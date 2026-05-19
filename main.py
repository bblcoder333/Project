import os
import csv
import traceback
from stages.ui_analysis     import load_model, analyze_ui, save_ui_data
from stages.rag_setup       import setup_knowledge_base
from stages.rag_retrieval   import retrieve_knowledge
from stages.test_generation import generate_test_cases, save_test_cases, append_to_master_csv

# Placeholders for Agents 3 and 4
# from stages.metamorphic  import generate_metamorphic_cases
# from stages.optimization import optimize_suite

IMAGES_DIR = "images/"
TOPICS_CSV = "design_topics.csv"

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

# ─── Load shared model once ───────────────────────────────────────────────────
print("\nLoading model...")
model, processor = load_model()
tokenizer = processor.tokenizer

# ─── Setup knowledge base once ────────────────────────────────────────────────
print("\nSetting up knowledge base...")
kb = setup_knowledge_base()

# ─── Process every image ──────────────────────────────────────────────────────
image_files = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.endswith((".png", ".jpg", ".jpeg"))
])
print(f"\n📁 Found {len(image_files)} images to process")

for image_file in image_files:
    full_path = os.path.join(IMAGES_DIR, image_file)
    screen_id = os.path.splitext(image_file)[0]
    topic     = id_to_topic.get(screen_id, "unknown")
    out_path  = os.path.join("outputs/ui_analysis", f"{screen_id}.json")

    # Skip if already processed
    if os.path.exists(out_path):
        print(f"⏭️  Skipping {image_file} — already processed")
        continue

    print(f"\n── Agent 1-4 Pipeline: {image_file} (topic: {topic}) ──")

    try:
        # ── Agent 1 — Perception ─────────────────────────────────────────
        ui_data         = analyze_ui(full_path, model, processor)
        ui_data["topic"] = topic
        save_ui_data(ui_data)
        print(f"✅ Agent 1 complete — UI data saved")

        # ── Agent 2 — RAG Retrieval ───────────────────────────────────────
        print("🔍 Retrieving knowledge...")
        retrieved_chunks = retrieve_knowledge(ui_data, kb)
        print(f"✅ Agent 2 (RAG) — {len(retrieved_chunks)} chunks retrieved")

        # ── Agent 2 — Test Case Generation ───────────────────────────────
        tc_data              = generate_test_cases(ui_data, retrieved_chunks, model, tokenizer)
        tc_data["screen_id"] = screen_id
        tc_data["topic"]     = topic
        print(f"✅ Agent 2 (Generation) — {len(tc_data.get('test_cases', []))} test cases")

        # ── Agent 3 — Metamorphic Testing (placeholder) ───────────────────
        # tc_data = generate_metamorphic_cases(tc_data)
        print("⚠️  Agent 3 — Metamorphic testing (coming next)")

        # ── Agent 4 — Optimization (placeholder) ─────────────────────────
        # tc_data = optimize_suite(tc_data)
        print("⚠️  Agent 4 — Optimization (coming next)")

        # ── Final Save ────────────────────────────────────────────────────
        save_test_cases(tc_data)
        append_to_master_csv(tc_data)
        print(f"✅ Pipeline complete for {screen_id}")

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Failed on {image_file}: {e}")
        continue

print("\n✅ All agents finished")