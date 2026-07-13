# test_agent4.py
#
# Runs Agent 4 (optimization/prioritization) over every metamorphic-relation
# CSV in outputs/metamorphic_relations/, using whichever text model you
# specify with --model. Universal across all model configs, same pattern as
# benchmark_pipeline.py.
#
# Usage:
#   python test_agent4.py --model qwen       (run from .venv)
#   python test_agent4.py --model phi35      (run from .venv-vision)
#   python test_agent4.py --model internvl   (run from .venv-vision)

import argparse
import csv
import importlib
import os

from stages.optimization import (
    optimize_metamorphic_relations,
    save_optimized_mr_data,
    append_optimized_mr_to_master,
)

# Maps --model name -> the module that provides load_text_model() for that
# model's text component. Must match the llm_module entries in
# benchmark_pipeline.py's MODELS list, since Agent 4 always uses the same
# text model as Agents 2 and 3 for a given model config.
LLM_MODULES = {
    "qwen":     "stages.test_generation",
    "phi35":    "stages.text_model_phi",
    "internvl": "stages.text_model_internlm",
}

parser = argparse.ArgumentParser(description="Run Agent 4 standalone against saved Agent 3 output.")
parser.add_argument(
    "--model",
    required=True,
    choices=list(LLM_MODULES.keys()),
    help=f"Which model's text component to use. Options: {list(LLM_MODULES.keys())}",
)
args = parser.parse_args()

llm_module = importlib.import_module(LLM_MODULES[args.model])
load_text_model = llm_module.load_text_model
warmup_fn = getattr(llm_module, "warmup", None)

# ─── Clear master ─────────────────────────────────────────────────────────────
master_path = "outputs/optimized_relations_master.csv"
if os.path.exists(master_path):
    os.remove(master_path)
    print(f"🗑️  Cleared {master_path}")

# ─── Load model once ──────────────────────────────────────────────────────────
print(f"\nLoading text model (Agent 4, model={args.model})...")
text_model, text_tokenizer = load_text_model()
if warmup_fn:
    warmup_fn(text_model, text_tokenizer)

# ─── Process every MR CSV ─────────────────────────────────────────────────────
mr_dir = "outputs/metamorphic_relations"
for filename in sorted(os.listdir(mr_dir)):
    if not filename.endswith("_MR.csv"):
        continue
    screen_id = filename.replace("_MR.csv", "")
    csv_path  = os.path.join(mr_dir, filename)
    mr_data   = {"screen_id": screen_id, "topic": "unknown", "metamorphic_relations": []}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mr_data["topic"] = row.get("topic", "unknown")
            mr_data["metamorphic_relations"].append({
                "mr_id":               row["mr_id"],
                "source_tc_id":        row["source_tc_id"],
                "mr_category":         row["mr_category"],
                "transformation":      row["transformation"],
                "follow_up_steps":     row["follow_up_steps"],
                "follow_up_test_data": row["follow_up_test_data"],
                "expected_relation":   row["expected_relation"],
                "confidence":          row["confidence"],
            })
    print(f"\n── Agent 4 [{args.model}]: {screen_id} ({len(mr_data['metamorphic_relations'])} MRs) ──")
    opt_data = optimize_metamorphic_relations(mr_data, text_model, text_tokenizer)
    save_optimized_mr_data(opt_data)
    append_optimized_mr_to_master(opt_data)
    print(f"--- Optimization decisions for {screen_id} ---")
    for opt in opt_data.get("optimized_relations", []):
        print(f"  {opt['mr_id']} | {opt['mr_category']} | {opt['decision']} | {opt['reason']}")

print(f"\n✅ Agent 4 done (model={args.model})")