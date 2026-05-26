import os
import json
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

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
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    return model, processor

def analyze_ui(image_path: str, model, processor) -> dict:
    image = Image.open(image_path).convert("RGB")
    screen_id = os.path.splitext(os.path.basename(image_path))[0]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": SCREENSHOT_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    output = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]

    return {
        "screen_id": screen_id,
        "description": output.strip(),
    }

def save_ui_data(data: dict, out_dir: str = "outputs/ui_analysis"):
    os.makedirs(out_dir, exist_ok=True)
    screen_id = data["screen_id"]

    # Save .json
    json_path = os.path.join(out_dir, f"{screen_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Save .txt
    txt_path = os.path.join(out_dir, f"{screen_id}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Screen ID: {screen_id}\n")
        f.write("=" * 60 + "\n\n")
        f.write(data.get("description", ""))

    return json_path, txt_path