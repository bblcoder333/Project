# test_agent3.py
#
# Runs Agent 3 (metamorphic relation generation) over every test-case CSV in
# outputs/testcases/, using whichever text model you specify with --model.
# Universal across all model configs, same pattern as benchmark_pipeline.py.
#
# Usage:
#   python test_agent3.py --model qwen       (run from .venv)
#   python test_agent3.py --model phi35      (run from .venv-vision)
#   python test_agent3.py --model internvl   (run from .venv-vision)

import argparse
import csv
import importlib
import os

from stages.metamorphic_testing import generate_metamorphic_relations, save_mr_data, append_mr_to_master_csv

# Maps --model name -> the module that provides load_text_model() for that
# model's text component. Must match the llm_module entries in
# benchmark_pipeline.py's MODELS list, since Agent 3 always uses the same
# text model as Agents 2 and 4 for a given model config.
LLM_MODULES = {
    "qwen":     "stages.test_generation",
    "phi35":    "stages.text_model_phi",
    "internvl": "stages.text_model_internlm",
}

parser = argparse.ArgumentParser(description="Run Agent 3 standalone against saved Agent 2 output.")
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
master_path = "outputs/metamorphic_relations_master.csv"
if os.path.exists(master_path):
    os.remove(master_path)
    print(f"🗑️  Cleared {master_path}")

# ─── Load model once ──────────────────────────────────────────────────────────
print(f"\nLoading text model (Agent 3, model={args.model})...")
text_model, text_tokenizer = load_text_model()
if warmup_fn:
    warmup_fn(text_model, text_tokenizer)

csv_dir = "outputs/testcases"
for filename in sorted(os.listdir(csv_dir)):
    if not filename.endswith(".csv"):
        continue
    screen_id = os.path.splitext(filename)[0]
    csv_path  = os.path.join(csv_dir, filename)
    tc_data   = {"screen_id": screen_id, "topic": "unknown", "test_cases": []}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tc_data["topic"] = row.get("topic", "unknown")
            tc_data["test_cases"].append({
                "test_case_id": row["test_case_id"],
                "title":        row["title"],
                "type":         row["type"],
                "test_data":    [row["test_data"]] if row["test_data"] else [],
                "steps":        [s.strip() for s in row["test_steps"].split(",")],
                "expected":     [row["expected_result"]],
            })
    print(f"\n── Agent 3 [{args.model}]: {screen_id} ({len(tc_data['test_cases'])} source TCs) ──")
    mr_data = generate_metamorphic_relations(tc_data, text_model, text_tokenizer)
    save_mr_data(mr_data)
    append_mr_to_master_csv(mr_data)
    print(f"--- MRs generated for {screen_id} ---")
    for mr in mr_data["metamorphic_relations"]:
        print(f"  {mr['mr_id']} | {mr['source_tc_id']} | {mr['mr_category']} | {mr['transformation']}")

print(f"\n✅ Agent 3 done (model={args.model})")