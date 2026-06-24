# test_agent3.py
import csv
import os
from stages.metamorphic_testing import generate_metamorphic_relations, save_mr_data, append_mr_to_master_csv

# Clear master
master_path = "outputs/metamorphic_relations_master.csv"
if os.path.exists(master_path):
    os.remove(master_path)

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

    print(f"\n── Agent 3: {screen_id} ({len(tc_data['test_cases'])} source TCs) ──")
    mr_data = generate_metamorphic_relations(tc_data)
    save_mr_data(mr_data)
    append_mr_to_master_csv(mr_data)

    print(f"--- MRs generated for {screen_id} ---")
    for mr in mr_data["metamorphic_relations"]:
        print(f"  {mr['mr_id']} | {mr['source_tc_id']} | {mr['mr_category']} | {mr['transformation']}")

print("\n✅ Agent 3 done")