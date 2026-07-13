# test_agent2.py
#
# Chains Agent 1 -> Agent 2 on ONE real image, so you can confirm the text
# model for a given config loads correctly and actually produces parseable
# test-case JSON, before committing to the full 5-run benchmark.
# Universal across all model configs, same pattern as benchmark_pipeline.py.
#
# Usage:
#   python test_agent2_only.py --model qwen       (run from .venv)
#   python test_agent2_only.py --model phi35      (run from .venv-vision)
#   python test_agent2_only.py --model internvl   (run from .venv-vision)

import argparse
import importlib
import os
import json

from stages.test_generation import generate_test_cases, save_test_cases

# Maps --model name -> the module providing load_model()/warmup()/analyze_ui()
# (vision) and load_text_model()/warmup() (text). Must match the vlm_module /
# llm_module entries in benchmark_pipeline.py's MODELS list.
VLM_MODULES = {
    "qwen":     "stages.ui_analysis",
    "phi35":    "stages.ui_analysis_phi",
    "internvl": "stages.ui_analysis_internvl",
}
LLM_MODULES = {
    "qwen":     "stages.test_generation",
    "phi35":    "stages.text_model_phi",
    "internvl": "stages.text_model_internlm",
}

parser = argparse.ArgumentParser(description="Smoke test Agent 1 + Agent 2 chained on one real image.")
parser.add_argument(
    "--model",
    required=True,
    choices=list(VLM_MODULES.keys()),
    help=f"Which model config to test. Options: {list(VLM_MODULES.keys())}",
)
args = parser.parse_args()

vlm_module = importlib.import_module(VLM_MODULES[args.model])
load_vision_model = vlm_module.load_model
analyze_ui = vlm_module.analyze_ui
warmup_vision = getattr(vlm_module, "warmup", None)

llm_module = importlib.import_module(LLM_MODULES[args.model])
load_text_model = llm_module.load_text_model
warmup_text = getattr(llm_module, "warmup", None)

# ── Agent 1 — get a real UI description to feed into Agent 2 ─────────────────
print(f"=== Loading Agent 1 ({args.model}) ===")
vision_model, processor = load_vision_model()
if warmup_vision:
    warmup_vision(vision_model, processor)

image_files = [f for f in os.listdir("images/") if f.endswith((".png", ".jpg", ".jpeg"))]
if not image_files:
    raise SystemExit(
        "❌ No images found in images/. Make sure your ENRICO screenshots "
        "are uploaded to the images/ folder before running this test."
    )

test_image = os.path.join("images/", image_files[0])
print(f"\nRunning Agent 1 on: {test_image}")
ui_data = analyze_ui(test_image, vision_model, processor)
ui_data["screen_id"] = os.path.splitext(os.path.basename(test_image))[0]
ui_data["topic"] = "unknown"
print("✅ Agent 1 description (first 300 chars):")
print(ui_data["description"][:300])

# Free the vision model's VRAM before loading the text model — avoids OOM
# on GPUs that can't comfortably hold both 7B+ models at once.
import gc
import torch
del vision_model, processor
gc.collect()
torch.cuda.empty_cache()
print("\n🗑️  Freed vision model from VRAM")

# ── Agent 2 — generate test cases with the text model ────────────────────────
print(f"\n=== Loading Agent 2 ({args.model}) ===")
text_model, text_tokenizer = load_text_model()
if warmup_text:
    warmup_text(text_model, text_tokenizer)

print(f"\nRunning Agent 2 on screen: {ui_data['screen_id']}")
tc_data = generate_test_cases(ui_data, text_model, text_tokenizer)
tc_data["screen_id"] = ui_data["screen_id"]
tc_data["topic"] = ui_data["topic"]

test_cases = tc_data.get("test_cases", [])
print(f"\n✅ Agent 2 generated {len(test_cases)} test cases")

if not test_cases:
    print("⚠️  Zero test cases were generated — check the raw model output / JSON parsing above.")
else:
    print("\n--- First test case (sanity check) ---")
    print(json.dumps(test_cases[0], indent=2))

save_test_cases(tc_data, out_dir=f"outputs/smoke_test_{args.model}/testcases")
print(f"\n✅ Agent 1 + Agent 2 smoke test passed (model={args.model})")