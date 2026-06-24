import os
import csv
import time
import traceback
from codecarbon import EmissionsTracker
from stages.ui_analysis        import load_model, analyze_ui, save_ui_data
from stages.test_generation    import load_text_model, generate_test_cases, save_test_cases, append_to_master_csv
from stages.metamorphic_testing import (
    generate_metamorphic_relations,
    save_mr_data,
    append_mr_to_master_csv,
)
from stages.optimization import (
    optimize_metamorphic_relations,
    save_optimized_mr_data,
    append_optimized_mr_to_master,
)

IMAGES_DIR = "images/"
TOPICS_CSV = "design_topics.csv"

# ─── Emissions log columns ────────────────────────────────────────────────────
EMISSIONS_LOG = "outputs/emissions_log.csv"
EMISSIONS_COLUMNS = [
    "screen_id", "agent", "energy_kwh", "emissions_kg_co2", "duration_seconds"
]

def _init_emissions_log():
    os.makedirs("outputs", exist_ok=True)
    if os.path.exists(EMISSIONS_LOG):
        os.remove(EMISSIONS_LOG)
    with open(EMISSIONS_LOG, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=EMISSIONS_COLUMNS).writeheader()

def _log_emissions(screen_id: str, agent: str, tracker: EmissionsTracker, duration_s: float):
    """Stop tracker and append one row to the emissions log."""
    emissions_kg = tracker.stop()
    energy_kwh   = tracker._total_energy.kWh if hasattr(tracker, "_total_energy") else 0.0
    with open(EMISSIONS_LOG, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=EMISSIONS_COLUMNS).writerow({
            "screen_id":        screen_id,
            "agent":            agent,
            "energy_kwh":       round(energy_kwh, 6),
            "emissions_kg_co2": round(emissions_kg or 0.0, 8),
            "duration_seconds": round(duration_s, 2),
        })
    print(f"   🌱 {agent} — {round(emissions_kg or 0.0, 8)} kg CO2 | "
          f"{round(energy_kwh, 6)} kWh | {round(duration_s, 2)}s")

# ─── Load topic mapping ───────────────────────────────────────────────────────
id_to_topic = {}
if os.path.exists(TOPICS_CSV):
    with open(TOPICS_CSV, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                id_to_topic[row[0].strip()] = row[1].strip()
    print(f"✅ Loaded {len(id_to_topic)} topic mappings")
else:
    print(f"⚠️  {TOPICS_CSV} not found — topics will be 'unknown'")

# ─── Clear master CSVs and init emissions log ─────────────────────────────────
master_files = [
    "outputs/testcases_master.csv",
    "outputs/metamorphic_relations_master.csv",
    "outputs/optimized_relations_master.csv",
]
for master_path in master_files:
    if os.path.exists(master_path):
        os.remove(master_path)
        print(f"🗑️  Cleared {master_path} — rebuilding fresh")

_init_emissions_log()
print("🌱 Emissions log initialized")

# ─── Load vision model once (Agent 1) ────────────────────────────────────────
print("\nLoading vision model (Agent 1)...")
vision_model, processor = load_model()

# ─── Load text model once (Agents 2, 3, 4 share it) ─────────────────────────
print("\nLoading text model (Agents 2 / 3 / 4)...")
text_model, text_tokenizer = load_text_model()

# ─── Process every image ─────────────────────────────────────────────────────
image_files = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.endswith((".png", ".jpg", ".jpeg"))
])
print(f"\n📁 Found {len(image_files)} images to process")

for image_file in image_files:
    full_path = os.path.join(IMAGES_DIR, image_file)
    screen_id = os.path.splitext(image_file)[0]
    topic     = id_to_topic.get(screen_id, "unknown")

    print(f"\n── Agent 1-4 Pipeline: {image_file} (topic: {topic}) ──")

    try:
        # ── Agent 1 — Perception ─────────────────────────────────────────
        tracker = EmissionsTracker(
            project_name=f"agent1_{screen_id}",
            output_dir="outputs",
            log_level="error",
            save_to_file=False,
        )
        tracker.start()
        _t0 = time.time()
        ui_data          = analyze_ui(full_path, vision_model, processor)
        ui_data["topic"] = topic
        json_path, txt_path = save_ui_data(ui_data)
        _log_emissions(screen_id, "Agent1_Perception", tracker, time.time() - _t0)
        print(f"✅ Agent 1 complete — UI data saved")

        # ── Agent 2 — Test Case Generation ───────────────────────────────
        tracker = EmissionsTracker(
            project_name=f"agent2_{screen_id}",
            output_dir="outputs",
            log_level="error",
            save_to_file=False,
        )
        tracker.start()
        _t0 = time.time()
        tc_data              = generate_test_cases(ui_data, text_model, text_tokenizer)
        tc_data["screen_id"] = screen_id
        tc_data["topic"]     = topic
        _log_emissions(screen_id, "Agent2_Generation", tracker, time.time() - _t0)
        print(f"✅ Agent 2 (Generation) — {len(tc_data.get('test_cases', []))} test cases")

        save_test_cases(tc_data)
        append_to_master_csv(tc_data)

        # ── Agent 3 — Metamorphic Testing (rule-based) ───────────────────
        tracker = EmissionsTracker(
            project_name=f"agent3_{screen_id}",
            output_dir="outputs",
            log_level="error",
            save_to_file=False,
        )
        tracker.start()
        _t0 = time.time()
        ui_description = ui_data.get("description", "")
        mr_data = generate_metamorphic_relations(tc_data, ui_description)
        _log_emissions(screen_id, "Agent3_Metamorphic", tracker, time.time() - _t0)
        print(f"✅ Agent 3 (Metamorphic) — {len(mr_data.get('metamorphic_relations', []))} MRs")

        save_mr_data(mr_data)
        append_mr_to_master_csv(mr_data)

        # ── Agent 4 — Optimization ────────────────────────────────────────
        tracker = EmissionsTracker(
            project_name=f"agent4_{screen_id}",
            output_dir="outputs",
            log_level="error",
            save_to_file=False,
        )
        tracker.start()
        _t0 = time.time()
        opt_data = optimize_metamorphic_relations(mr_data, text_model, text_tokenizer)
        _log_emissions(screen_id, "Agent4_Optimization", tracker, time.time() - _t0)
        print(f"✅ Agent 4 (Optimization) — {len(opt_data.get('optimized_relations', []))} optimized MRs")

        save_optimized_mr_data(opt_data)
        append_optimized_mr_to_master(opt_data)

        print(f"✅ Pipeline complete for {screen_id}")

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Failed on {image_file}: {e}")
        continue

print("\n✅ All agents finished")
print(f"🌱 Emissions log saved → {EMISSIONS_LOG}")