import csv
import os
from stages.optimization import (
    optimize_metamorphic_relations,
    save_optimized_mr_data,
    append_optimized_mr_to_master,
)
from stages.test_generation import load_text_model

# ─── Clear master ─────────────────────────────────────────────────────────────
master_path = "outputs/optimized_relations_master.csv"
if os.path.exists(master_path):
    os.remove(master_path)
    print("🗑️  Cleared optimized master CSV")

# ─── Load model once ──────────────────────────────────────────────────────────
print("\nLoading text model (Agent 4)...")
text_model, text_tokenizer = load_text_model()

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

    print(f"\n── Agent 4: {screen_id} ({len(mr_data['metamorphic_relations'])} MRs) ──")
    opt_data = optimize_metamorphic_relations(mr_data, text_model, text_tokenizer)
    save_optimized_mr_data(opt_data)
    append_optimized_mr_to_master(opt_data)

    print(f"--- Optimization decisions for {screen_id} ---")
    for opt in opt_data.get("optimized_relations", []):
        print(f"  {opt['mr_id']} | {opt['mr_category']} | {opt['decision']} | {opt['reason']}")

print("\n✅ Agent 4 done")