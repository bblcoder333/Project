"""
stages/ui_analysis_internvl.py
Agent 1 — UI Perception using InternVL2-8B (OpenGVLab / Shanghai AI Lab)

Drop-in replacement for stages/ui_analysis.py / stages/ui_analysis_phi.py.

IMPORTANT — this model's API is genuinely different from Qwen2-VL and
Phi-3.5-vision, not just a different model string:
  - Uses AutoModel (NOT AutoModelForCausalLM / Qwen2VLForConditionalGeneration)
  - Image preprocessing is DIY: InternVL2 does NOT use an AutoProcessor.
    You build your own torchvision transform + "dynamic tiling" preprocessing
    (the dynamic_preprocess()/load_image() functions below are the standard
    boilerplate straight from OpenGVLab's model card — do not simplify this,
    the tiling step measurably affects output quality on high-res screenshots).
  - Generation happens via a custom model.chat(tokenizer, pixel_values,
    question, generation_config) method, NOT model.generate(**inputs).
  - trust_remote_code=True required.
  - use_flash_attn=False is REQUIRED unless flash_attn is actually installed —
    OpenGVLab's example code defaults this to True, which will raise an
    ImportError on a machine without flash_attn (same class of problem you
    hit with Phi-3.5-vision, different symptom).
  - Given InternVL2-8B is from the same mid-2024 generation of custom
    trust_remote_code releases as Phi-3.5, treat a modern transformers
    version as a likely incompatibility risk. If you hit a cache/attribute
    error similar to the Phi DynamicCache issue, pin transformers to
    ~4.46.x (or whatever version OpenGVLab's own examples were tested
    against — check the model card / GitHub issues for the exact version
    reported working, since this can shift over time) in its own venv,
    the same way you did for Phi.
"""

import os
import json

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

INTERNVL_MODEL_ID = "OpenGVLab/InternVL2-8B"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ─── Prompt (identical task/format to your Qwen and Phi versions) ─────────────

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


# ─── Image preprocessing — standard InternVL2 boilerplate ───────────────────
# This dynamic-tiling preprocessing is what InternVL2 was trained with; do not
# swap it for a plain resize, it measurably affects quality on real screenshots.

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_tensor(image_path: str, input_size: int = 448, max_num: int = 12):
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model():
    """Load InternVL2-8B and its tokenizer. Returns (model, tokenizer) —
    same (model, processor)-shaped tuple your pipeline expects, just with
    'processor' actually being the tokenizer for this model family."""
    print(f"   Loading InternVL2-8B from {INTERNVL_MODEL_ID}...")
    model = AutoModel.from_pretrained(
        INTERNVL_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,  # avoids requiring the flash_attn package
        trust_remote_code=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        INTERNVL_MODEL_ID,
        trust_remote_code=True,
        use_fast=False,
    )
    print("   ✅ InternVL2-8B loaded")
    return model, tokenizer


def warmup(model, tokenizer):
    """Single throwaway inference pass to warm up GPU caches before timed runs."""
    print("   🔥 Warming up InternVL2-8B...")
    dummy_image = Image.new("RGB", (448, 448), color=(0, 0, 0))
    transform = build_transform(input_size=448)
    pixel_values = transform(dummy_image).unsqueeze(0).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=5, do_sample=False)
    with torch.no_grad():
        model.chat(tokenizer, pixel_values, "<image>\nDescribe this briefly.", generation_config)
    print("   ✅ Warmup complete")


def analyze_ui(image_path: str, model, tokenizer) -> dict:
    """Run Agent 1 perception on a single screenshot using InternVL2-8B."""
    screen_id = os.path.splitext(os.path.basename(image_path))[0]

    pixel_values = load_image_tensor(image_path, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=4096, do_sample=False)

    question = f"<image>\n{SCREENSHOT_PROMPT}"

    with torch.no_grad():
        response = model.chat(tokenizer, pixel_values, question, generation_config)

    return {
        "screen_id": screen_id,
        "description": response.strip(),
    }


def save_ui_data(data: dict, out_dir: str = "outputs/ui_analysis") -> tuple:
    """Save UI description to JSON and TXT — identical to the other versions."""
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