import os
import json
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# ─── PROMPT ───────────────────────────────────────────────────────────────────
SCREENSHOT_PROMPT = """
You are a senior mobile UI/UX analyst and frontend design reviewer.

You are analyzing ONE FINAL IMPLEMENTED MOBILE APP SCREENSHOT.
Describe the screen in detail for a developer who cannot see the image.

ABSOLUTE RULES:
- Always describe the UI from TOP TO BOTTOM.
- Always talk in terms of UI components:
  status bar, app bar, title, subtitle, image, icon, text field,
  card, list row, toggle, checkbox, radio button, chip,
  CTA button, floating action button, tab bar, bottom nav, footer.
- If any text is readable, include it verbatim.
- Never skip ANY region of the screen.
- Include ALL visible UI elements even if partially visible.
- Include spacing, alignment, padding, and layout structure.
- Mention approximate component positioning:
  left-aligned, centered, right-aligned, full-width, floating, stacked, grid-based, etc.

VISUAL DESIGN ANALYSIS:
For important UI components, describe:
- Background colors
- Text colors
- Button colors
- Accent colors
- Border colors
- Shadows/elevation
- Rounded corners / border radius
- Typography style
- Estimated font family/style if recognizable
  (e.g., San Francisco, Roboto, Inter, Material Design style)
- Font weight and hierarchy
  (bold heading, medium subtitle, small caption, etc.)
- Contrast and readability
- Spacing and visual density
- Icon style (outlined, filled, minimal, skeuomorphic, etc.)

INTERACTION + POINTER ANALYSIS:
- Identify likely tappable/clickable areas.
- Mention touch targets and interactive affordances.
- Describe navigation indicators:
  chevrons, arrows, highlighted tabs, active states, toggles.
- Mention gesture hints if visible.
- Identify the primary CTA and secondary actions.
- Mention hover/pointer style clues if applicable.

BACKGROUND + CONTAINER ANALYSIS:
- Describe the overall background:
  solid color, gradient, image, blur, glassmorphism, etc.
- Mention cards, containers, sections, and grouping behavior.
- Describe layering and hierarchy between components.

OUTPUT STRUCTURE (use these headings exactly):

A) Screen purpose and type:
B) Layout top to bottom:
C) Visual design system:
D) Component inventory (grouped):
E) Primary actions + navigation:
F) Interaction and usability observations:

Write 16-26 sentences total.
Be compact but highly detailed and information-dense.

End with this exact recap format:

- Screen type/purpose:
- Primary CTA(s):
- Main navigation pattern:
- Dominant colors:
- Typography style:
- Key interactive components:


Provide your analysis inside these tags:
[DESCRIPTION]
(Put your A-F analysis and recap here)

[ELEMENTS]
(Provide a bulleted list of just the interactive components found)
"""

# ─── MODEL LOAD ───────────────────────────────────────────────────────────────
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
    image     = Image.open(image_path).convert("RGB")
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

    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    output = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]

    # Parse into description + elements
    parts       = output.split("[ELEMENTS]")
    description = parts[0].replace("[DESCRIPTION]", "").strip()
    elements    = parts[1].strip() if len(parts) > 1 else ""

    return {
        "screen_id":           screen_id,
        "description":         description,
        "structured_elements": elements,
    }


def save_ui_data(data: dict, out_dir: str = "outputs/ui_analysis"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{data['screen_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  💾 UI data saved → {path}")
    return path