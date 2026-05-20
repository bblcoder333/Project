import os
import json
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


SCREENSHOT_PROMPT = """
You are a senior mobile UI/UX analyst, accessibility auditor, and frontend design reviewer.

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
- Spacing and visual density
- Icon style (outlined, filled, minimal, skeuomorphic, etc.)

TYPOGRAPHY & FONT ANALYSIS (analyze every text element):
For EACH distinct text element or text group, describe:
- Estimated font family/style
  (e.g., San Francisco, Roboto, Inter, Material Design style, serif, sans-serif)
- Font weight
  (e.g., thin ~100, light ~300, regular ~400, medium ~500, semibold ~600, bold ~700, extrabold ~800)
- Estimated font size in sp
  (e.g., display ~34sp, heading ~24sp, title ~20sp, body ~16sp, caption ~12sp, label ~11sp)
- Letter spacing / tracking
  (tight, normal, wide, all-caps with tracking)
- Line height / leading
  (compact, normal, relaxed)
- Text alignment
  (left, center, right, justified)
- Text decoration
  (underline, strikethrough, none)
- Text truncation or overflow behavior if visible
  (ellipsis, clipping, wrapping)
- Font hierarchy level
  (H1, H2, H3, body, caption, label, helper text, placeholder)

TYPOGRAPHY ACCESSIBILITY AUDIT (per text element):
Evaluate each text element against these accessibility standards:
- Minimum font size: flag any text estimated below 12sp as potentially inaccessible
- Font weight readability: flag thin fonts (weight ~100-200) on low-contrast backgrounds
- Text contrast:
  * Normal text (<18sp or <14sp bold): needs minimum 4.5:1 contrast ratio
  * Large text (≥18sp or ≥14sp bold): needs minimum 3:1 contrast ratio
  * Estimate contrast as: Pass, Likely Pass, Likely Fail, or Fail based on visual inspection
- Line length readability: flag lines exceeding ~75 characters as potentially hard to read
- Touch target size for text links: minimum 44x44dp recommended
- All-caps text: flag if used for body text (reduces readability for dyslexic users)
- Placeholder text contrast: placeholder text is often low contrast — flag if appears grey on white
- Dynamic text support: note if layout appears to support text scaling
- Text on image/gradient backgrounds: flag if text appears over busy backgrounds without overlay

BACKGROUND & CONTAINER ANALYSIS (analyze every background layer):
For EACH distinct background region or container, describe:
- Background type:
  (solid color, linear gradient, radial gradient, image, blur/frosted glass,
   glassmorphism, mesh gradient, pattern, dark overlay, transparent, none)
- Background color or gradient:
  (describe start/end colors and direction for gradients,
   e.g., top-left blue #1A73E8 → bottom-right purple #6C47FF)
- Overlay or scrim:
  (describe any dark/light overlay on top of images or gradients,
   e.g., 40% black scrim over hero image)
- Card and container backgrounds:
  (white card on grey background, elevated surface, flat surface,
   border only, outlined card, filled card)
- Elevation and shadow:
  (no shadow, subtle shadow, strong drop shadow, inner shadow,
   Material elevation level if recognizable: 0dp, 2dp, 4dp, 8dp, 16dp)
- Border and dividers:
  (hairline dividers, full borders, dashed borders, no borders,
   estimated border color and thickness)
- Corner radius per container:
  (sharp corners 0dp, slight rounding ~4dp, medium ~8dp,
   large ~16dp, pill-shaped ~50dp, fully circular)
- Transparency and layering:
  (describe if any elements appear semi-transparent or layered over each other)
- Background pattern or texture:
  (flat, subtle noise, grain, geometric pattern, illustration)

BACKGROUND ACCESSIBILITY AUDIT (per background region):
Evaluate each background against these accessibility standards:
- Text over background contrast:
  * Flag any text placed directly over busy images without sufficient overlay
  * Flag gradient backgrounds where text contrast may vary across the element
  * Estimate contrast as: Pass, Likely Pass, Likely Fail, or Fail
- Background color and cognitive load:
  * Flag highly saturated or bright backgrounds that may cause visual fatigue
  * Flag pure white (#FFFFFF) backgrounds with no surface differentiation
  * Flag pure black (#000000) backgrounds with harsh contrast
- Dark mode compatibility clues:
  * Note if the background palette appears dark-mode friendly or strictly light
- Motion and animation hints:
  * Note any animated background indicators (shimmer, pulse, loading states)
- Focus visibility on backgrounds:
  * Flag if interactive elements on this background would have poor focus ring visibility

INTERACTION + POINTER ANALYSIS:
- Identify likely tappable/clickable areas.
- Mention touch targets and interactive affordances.
- Describe navigation indicators:
  chevrons, arrows, highlighted tabs, active states, toggles.
- Mention gesture hints if visible.
- Identify the primary CTA and secondary actions.
- Mention hover/pointer style clues if applicable.

OUTPUT STRUCTURE — YOU MUST USE THESE EXACT HEADINGS FOR EVERY SCREEN NO EXCEPTIONS:

A) Screen purpose and type:
B) Layout top to bottom:
C) Visual design system:
D) Typography analysis (per text element):
E) Typography accessibility audit:
F) Background and container analysis (per region):
G) Background accessibility audit:
H) Component inventory (grouped):
I) Primary actions + navigation:
J) Interaction and usability observations:

CRITICAL RULES FOR OUTPUT:
- You MUST output ALL sections A through J every time.
- You MUST use the exact letter headings shown above.
- Do NOT merge sections into paragraphs.
- Do NOT skip any section even if the screen is simple.
- Each section must be on its own line starting with the letter heading.
- Write 24-35 sentences total spread across all sections.

Write 24-35 sentences total.
Be compact but highly detailed and information-dense.

End with this exact recap format:

- Screen type/purpose:
- Primary CTA(s):
- Main navigation pattern:
- Dominant colors:
- Typography system: (font family, size range, weight range)
- Background system: (background types used, dominant surface color)
- Accessibility flags: (list any potential issues found)
- Key interactive components:

Provide your analysis inside these tags:
[DESCRIPTION]
(Put your A-J analysis and recap here)

[ELEMENTS]
(Provide a bulleted list of just the interactive components found)
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

    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    output = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]

    # Retry with more tokens if output was truncated before [ELEMENTS]
    if "[ELEMENTS]" not in output:
        print("  ⚠️  Output truncated — retrying with more tokens...")
        generated_ids = model.generate(**inputs, max_new_tokens=1500, do_sample=False)
        output = processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]

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
    screen_id = data["screen_id"]

    # Save .json
    json_path = os.path.join(out_dir, f"{screen_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  💾 UI data saved → {json_path}")

    # Save .txt — full readable description
    txt_path = os.path.join(out_dir, f"{screen_id}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Screen ID: {screen_id}\n")
        f.write(f"Topic: {data.get('topic', 'unknown')}\n")
        f.write("=" * 60 + "\n\n")
        f.write("DESCRIPTION:\n")
        f.write(data.get("description", "") + "\n\n")
        f.write("=" * 60 + "\n\n")
        f.write("STRUCTURED ELEMENTS:\n")
        f.write(data.get("structured_elements", "") + "\n")
    print(f"  💾 UI text saved  → {txt_path}")

    return json_path, txt_path