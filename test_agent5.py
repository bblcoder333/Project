# test_agent5.py
#
# Runs Agent 5 (execution tiering + energy savings estimate) over every
# optimized-relations CSV in outputs/optimized_relations/, using whichever
# text model you specify with --model. Universal across all model configs,
# same pattern as test_agent3.py / test_agent4.py / benchmark_pipeline.py.
#
# Usage:
#   python test_agent5.py --model qwen       (run from .venv)
#   python test_agent5.py --model phi35      (run from .venv-vision)
#   python test_agent5.py --model internvl   (run from .venv-vision)

import argparse
import csv
import importlib
import os

from stages.optimization_reduction import (
    generate_reduced_suite,
    save_reduced_suite,
    append_reduced_to_master,
    append_savings_summary,
)

# Maps --model name -> the module that provides load_text_model() for that
# model's text component. Must match the llm_module entries in
# benchmark_pipeline.py's MODELS list, since Agent 5 always uses the same
# text model as Agents 2, 3, and 4 for a given model config.
LLM_MODULES = {
    "qwen":     "stages.test_generation",
    "phi35":    "stages.text_model_phi",
    "internvl": "stages.text_model_internlm",
}

parser = argparse.ArgumentParser(description="Run Agent 5 standalone against saved Agent 4 output.")
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

# ─── Clear masters ─────────────────────────────────────────────────────────────
for master_path in ["outputs/reduced_suite_master.csv", "outputs/energy_savings_summary.csv"]:
    if os.path.exists(master_path):
        os.remove(master_path)
        print(f"🗑️  Cleared {master_path}")

# ─── Load model once ──────────────────────────────────────────────────────────
print(f"\nLoading text model (Agent 5, model={args.model})...")
text_model, text_tokenizer = load_text_model()
if warmup_fn:
    warmup_fn(text_model, text_tokenizer)

# ─── Process every optimized-relations CSV ────────────────────────────────────
opt_dir = "outputs/optimized_relations"
if not os.path.isdir(opt_dir):
    raise SystemExit(
        f"❌ {opt_dir} not found. Run test_agent4.py (or the full benchmark) "
        "first so Agent 5 has Agent 4 output to reduce."
    )

overall_total = 0
overall_active = 0
overall_llm = 0
overall_clamped = 0
overall_regenerated = 0
overall_fallback = 0

for filename in sorted(os.listdir(opt_dir)):
    if not filename.endswith("_Prioritize_MR.csv"):
        continue

    screen_id = filename.replace("_Prioritize_MR.csv", "")
    csv_path  = os.path.join(opt_dir, filename)
    opt_data  = {"screen_id": screen_id, "topic": "unknown", "optimized_relations": []}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            opt_data["topic"] = row.get("topic", "unknown")
            opt_data["optimized_relations"].append({
                "mr_id":             row["mr_id"],
                "source_tc_id":      row["source_tc_id"],
                "mr_category":       row["mr_category"],
                "transformation":    row["transformation"],
                "expected_relation": row.get("expected_relation", ""),
                "decision":          row["decision"],
                "reason":            row["reason"],
            })

    print(f"\n── Agent 5 [{args.model}]: {screen_id} ({len(opt_data['optimized_relations'])} MRs from Agent 4) ──")
    reduced_data = generate_reduced_suite(opt_data, text_model, text_tokenizer)
    save_reduced_suite(reduced_data)
    append_reduced_to_master(reduced_data)
    append_savings_summary(reduced_data)

    s = reduced_data["summary"]
    overall_total       += s["total_mrs"]
    overall_active      += s["active_count"]
    overall_llm         += s["llm_tier_count"]
    overall_clamped     += s["clamped_tier_count"]
    overall_regenerated += s["regenerated_reasoning_count"]
    overall_fallback    += s["fallback_tier_count"]

    print(f"--- Tier breakdown for {screen_id} ---")
    for mr in reduced_data["reduced_relations"]:
        print(f"  {mr['mr_id']} | {mr['mr_category']} | {mr['decision']} -> {mr['execution_tier']} "
              f"(every {mr['cycle_frequency']} cycle(s)) — {mr['reasoning']}")

if overall_total:
    overall_reduction = round((1 - overall_active / overall_total) * 100, 2)
    print(f"\n📊 OVERALL: {overall_total} total MRs -> {overall_active} active "
          f"({overall_reduction}% suite reduction across all screens)")
    print(f"📊 Tier source: {overall_llm} from LLM, {overall_clamped} clamped, "
          f"{overall_regenerated} reasoning regenerated, {overall_fallback} from fallback")

print(f"\n✅ Agent 5 done (model={args.model})")