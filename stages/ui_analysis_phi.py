"""
stages/ui_analysis_phi.py
Agent 1 — UI Perception using Phi-3.5-vision-instruct (Microsoft)

Drop-in replacement for stages/ui_analysis.py.
Key differences from the Qwen2-VL version:
  - Uses AutoModelForCausalLM (not Qwen2VLForConditionalGeneration)
  - Image is embedded as <|image_1|> placeholder in the prompt string
  - processor() call takes (prompt_string, [image_list]) not separate inputs
  - trust_remote_code=True required
  - _attn_implementation='eager' used for broad GPU compatibility
    (change to 'flash_attention_2' if flash_attn is installed)
"""

import os
import json
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

PHI_VISION_MODEL_ID = "microsoft/Phi-3.5-vision-instruct"

# ─── Prompt (identical to your Qwen version — same task, same format) ─────────

SCREENSHOT_PROMPT = """
You are a senior mobile UI/UX analyst and accessibility auditor.

You are looking at ONE FINAL IMPLEMENTED MOBILE APP SCREENSHOT (a fully designed UI).
Describe the screen in detail for a developer who cannot see the image.

ABSOLUTE RULES:
- Always talk in terms of UI components:
  status bar, app bar, title, subtitle, image, icon, text field,
  card, list row, toggle, checkbox, radio button, chip,
  CTA button, floating action button, tab bar, bottom nav, footer.
- If any text is readable (titles, labels, button text, placeholders), include it verbatim.
- Do NOT skip ANY region of the screen — top to bottom.
- Do NOT skip the bottom area: capture CTA buttons and any footer/bottom nav if present.
- Include ALL visible UI elements even if partially visible.
- Mention approximate component positioning:
  left-aligned, centered, right-aligned, full-width, floating, stacked, grid-based.
- Do NOT repeat yourself.
- Do NOT skip any section even if the screen is simple.

FONT & TYPOGRAPHY (include for key text elements only):
- Estimated font family (e.g. Roboto, San Francisco, serif, sans-serif)
- Font weight (light/regular/medium/bold)
- Estimated font size (e.g. heading ~24sp, body ~16sp, caption ~12sp)
- Flag any text below 12sp or low contrast as an accessibility concern

OUTPUT STRUCTURE — YOU MUST USE THESE EXACT HEADINGS EVERY TIME NO EXCEPTIONS:

A) Screen purpose and type:
B) Layout top to bottom:
C) Component inventory (grouped):
D) Visual design (colors, typography, background):
E) Accessibility observations:
F) Primary actions + navigation:

Write 16-24 sentences total (compact but complete).
One paragraph per section maximum.

End with this exact bullet recap:
- Screen type/purpose:
- Key components:
- Primary CTA(s):
- Navigation/footer:
- Dominant colors:
- Typography: (font family, size range, weight range)
- Accessibility flags:
"""


def load_model():
    """Load Phi-3.5-vision-instruct VLM and its processor."""
    print(f"   Loading Phi-3.5-vision-instruct from {PHI_VISION_MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        PHI_VISION_MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",  # use 'flash_attention_2' if available
    )
    # num_crops=16 is recommended for single-frame image analysis
    processor = AutoProcessor.from_pretrained(
        PHI_VISION_MODEL_ID,
        trust_remote_code=True,
        num_crops=16,
    )
    print("   ✅ Phi-3.5-vision-instruct loaded")
    return model, processor


def warmup(model, processor):
    """
    Single throwaway inference pass to warm up GPU caches before timed runs.
    Uses a tiny 1x1 blank image so it completes near-instantly.
    """
    print("   🔥 Warming up Phi-3.5-vision-instruct...")
    dummy_image = Image.new("RGB", (1, 1), color=(0, 0, 0))
    messages = [{"role": "user", "content": "<|image_1|>\nDescribe this image briefly."}]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(prompt, [dummy_image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=5, do_sample=False,
                       eos_token_id=processor.tokenizer.eos_token_id)
    print("   ✅ Warmup complete")


def analyze_ui(image_path: str, model, processor) -> dict:
    """Run Agent 1 perception on a single screenshot using Phi-3.5-vision."""
    image = Image.open(image_path).convert("RGB")
    screen_id = os.path.splitext(os.path.basename(image_path))[0]

    # Phi-3.5-vision uses <|image_1|> as a placeholder inside the prompt string
    messages = [
        {
            "role": "user",
            "content": f"<|image_1|>\n{SCREENSHOT_PROMPT}",
        }
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(prompt, [image], return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    # Strip the input tokens from the output
    output = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return {
        "screen_id": screen_id,
        "description": output.strip(),
    }


def save_ui_data(data: dict, out_dir: str = "outputs/ui_analysis") -> tuple:
    """Save UI description to JSON and TXT — identical to the Qwen version."""
    os.makedirs(out_dir, exist_ok=True)
    screen_id = data["screen_id"]

    json_path = os.path.join(out_dir, f"{screen_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    txt_path = os.path.join(out_dir, f"{screen_id}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Screen ID: {screen_id}\n")
        f.write("=" * 60 + "\n\n")
        f.write(data.get("description", ""))

    return json_path, txt_path