# test_agent1.py
#
# Smoke test for Agent 1 (UI perception) only, on ONE real image from images/.
# Universal across all model configs, same pattern as benchmark_pipeline.py.
#
# Usage:
#   python test_agent1_only.py --model qwen       (run from .venv)
#   python test_agent1_only.py --model phi35      (run from .venv-vision)
#   python test_agent1_only.py --model internvl   (run from .venv-vision)

import argparse
import importlib
import os

# Maps --model name -> the module that provides load_model()/warmup()/
# analyze_ui()/save_ui_data() for that model's vision component. Must match
# the vlm_module entries in benchmark_pipeline.py's MODELS list.
VLM_MODULES = {
    "qwen":     "stages.ui_analysis",
    "phi35":    "stages.ui_analysis_phi",
    "internvl": "stages.ui_analysis_internvl",
}

parser = argparse.ArgumentParser(description="Smoke test Agent 1 (vision) on one real image.")
parser.add_argument(
    "--model",
    required=True,
    choices=list(VLM_MODULES.keys()),
    help=f"Which model's vision component to test. Options: {list(VLM_MODULES.keys())}",
)
args = parser.parse_args()

vlm_module = importlib.import_module(VLM_MODULES[args.model])
load_model = vlm_module.load_model
analyze_ui = vlm_module.analyze_ui
save_ui_data = vlm_module.save_ui_data
warmup_fn = getattr(vlm_module, "warmup", None)

print(f"=== Loading Agent 1 ({args.model}) ===")
vision_model, processor = load_model()
if warmup_fn:
    warmup_fn(vision_model, processor)

# pick one real image from your images/ folder
image_files = [f for f in os.listdir("images/") if f.endswith((".png", ".jpg", ".jpeg"))]
if not image_files:
    raise SystemExit(
        "❌ No images found in images/. Make sure your ENRICO screenshots "
        "are uploaded to the images/ folder before running this test."
    )

test_image = os.path.join("images/", image_files[0])
print(f"Testing on: {test_image}")

ui_data = analyze_ui(test_image, vision_model, processor)
print(ui_data["description"][:500])  # just print first 500 chars to sanity check

save_ui_data(ui_data, out_dir=f"outputs/smoke_test_{args.model}")
print(f"✅ Agent 1 smoke test passed (model={args.model})")