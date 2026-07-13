import argparse
import os
import csv
import time
import traceback
from codecarbon import EmissionsTracker

# ─── Config ───────────────────────────────────────────────────────────────────

IMAGES_DIR   = "images/"
TOPICS_CSV   = "design_topics.csv"

# Number of full pipeline runs per model (after the warmup pass)
NUM_RUNS = 5

# Cooldown between runs in seconds — lets GPU thermals/power stabilize
COOLDOWN_SECONDS = 60

# Models to benchmark — add more entries here as you test new SLMs.
# Each entry is a dict with:
#   "name"        → short label used in emissions CSV and output dirs
#   "vlm_module"  → dotted import path for the Agent 1 module
#   "llm_module"  → dotted import path for the shared text model loader
#
# IMPORTANT: qwen and phi35 require DIFFERENT transformers versions
# (qwen needs a modern transformers; Phi-3.5-vision's remote code only
# works on transformers<=4.46.x because Microsoft never patched it for
# newer cache APIs). You cannot have both installed in the same venv, so
# this script now runs ONE model per invocation — pick it with --model,
# and run this script once from each model's own venv.
MODELS = [
    {
        "name":       "qwen",
        "vlm_module": "stages.ui_analysis",
        "llm_module": "stages.test_generation",  # load_text_model lives here
    },
    {
        "name":       "phi35",
        "vlm_module": "stages.ui_analysis_phi",
        "llm_module": "stages.text_model_phi",
    },
    {
    "name":       "internvl",
    "vlm_module": "stages.ui_analysis_internvl",
    "llm_module": "stages.text_model_internlm",
    },
]

MODEL_NAMES = [m["name"] for m in MODELS]

# ─── CLI args ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Run the 4-agent pipeline benchmark for ONE model (run this "
                "script once per model, each from that model's own venv)."
)
parser.add_argument(
    "--model",
    required=True,
    choices=MODEL_NAMES,
    help=f"Which model config to benchmark. Options: {MODEL_NAMES}",
)
parser.add_argument(
    "--runs",
    type=int,
    default=NUM_RUNS,
    help=f"Number of timed runs (default: {NUM_RUNS})",
)
args = parser.parse_args()

NUM_RUNS = args.runs
model_cfg = next(m for m in MODELS if m["name"] == args.model)
model_name = model_cfg["name"]

# ─── Emissions log columns ────────────────────────────────────────────────────
# Each model gets its OWN log file so that benchmarking one model never
# overwrites or wipes out another model's results. summarize_emissions.py
# automatically finds and merges every emissions_log_*.csv file.
EMISSIONS_LOG     = f"outputs/emissions_log_{model_name}.csv"
EMISSIONS_COLUMNS = [
    "model_name", "run_number", "screen_id", "agent",
    "energy_kwh", "emissions_kg_co2", "total_emissions_kg_co2", "duration_seconds",
    "ram_power_w", "cpu_power_w", "gpu_power_w",
    "ram_energy_kwh", "cpu_energy_kwh", "gpu_energy_kwh",
    "region", "country_name", "country_iso_code",
]


def _init_emissions_log():
    os.makedirs("outputs", exist_ok=True)
    if os.path.exists(EMISSIONS_LOG):
        os.remove(EMISSIONS_LOG)
        print(f"🗑️  Cleared previous {EMISSIONS_LOG} — rebuilding fresh")
    with open(EMISSIONS_LOG, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=EMISSIONS_COLUMNS).writeheader()


def _init_model_output_dir(model_name: str):
    """
    Wipe outputs/{model_name}/ (all run_1..run_N folders and their master
    CSVs) before starting. Without this, a crashed/interrupted run leaves
    partial run folders behind, and append_to_master_csv()-style functions
    would silently append a SECOND copy of already-processed screens into
    those master CSVs on the next attempt — this guarantees a clean slate
    every time the benchmark is (re)started for a given model.
    """
    import shutil
    model_dir = os.path.join("outputs", model_name)
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        print(f"🗑️  Cleared previous outputs/{model_name}/ — rebuilding fresh")
    os.makedirs(model_dir, exist_ok=True)


def _log_emissions(model_name: str, run_number: int, screen_id: str,
                   agent: str, tracker: EmissionsTracker, duration_s: float):
    emissions_kg = tracker.stop()
    data         = tracker.final_emissions_data
    energy_kwh   = data.energy_consumed if data else 0.0

    with open(EMISSIONS_LOG, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=EMISSIONS_COLUMNS).writerow({
            "model_name":             model_name,
            "run_number":             run_number,
            "screen_id":              screen_id,
            "agent":                  agent,
            "energy_kwh":             round(energy_kwh, 6),
            "emissions_kg_co2":       round(emissions_kg or 0.0, 8),
            "total_emissions_kg_co2": round(emissions_kg or 0.0, 8),
            "duration_seconds":       round(duration_s, 2),
            "ram_power_w":            round(data.ram_power, 4)         if data else 0.0,
            "cpu_power_w":            round(data.cpu_power, 4)         if data else 0.0,
            "gpu_power_w":            round(data.gpu_power, 4)         if data else 0.0,
            "ram_energy_kwh":         round(data.ram_energy, 6)        if data else 0.0,
            "cpu_energy_kwh":         round(data.cpu_energy, 6)        if data else 0.0,
            "gpu_energy_kwh":         round(data.gpu_energy, 6)        if data else 0.0,
            "region":                 (data.region or "unknown")        if data else "unknown",
            "country_name":           (data.country_name or "unknown")  if data else "unknown",
            "country_iso_code":       (data.country_iso_code or "unknown") if data else "unknown",
        })

    print(f"   🌱 {agent} — {round(emissions_kg or 0.0, 8)} kg CO2 | "
          f"{round(energy_kwh, 6)} kWh | {round(duration_s, 2)}s | "
          f"GPU: {round(data.gpu_power, 2) if data else 0.0}W | "
          f"region: {data.region if data else 'unknown'}")


# ─── Dynamic import helper ────────────────────────────────────────────────────

def _import(module_path: str, attr: str):
    """Import a single attribute from a dotted module path."""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


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

# ─── Init emissions log (only this model's file) and collect image list ─────

_init_emissions_log()
print(f"🌱 Emissions log initialized → {EMISSIONS_LOG}")
_init_model_output_dir(model_name)

image_files = sorted([
    f for f in os.listdir(IMAGES_DIR)
    if f.endswith((".png", ".jpg", ".jpeg"))
])
print(f"\n📁 Found {len(image_files)} images to process")

# ─── Import stage functions shared across all models ─────────────────────────
# These don't change between models — only the model/tokenizer passed in changes.

from stages.test_generation     import generate_test_cases, save_test_cases, append_to_master_csv
from stages.metamorphic_testing import generate_metamorphic_relations, save_mr_data, append_mr_to_master_csv
from stages.optimization        import optimize_metamorphic_relations, save_optimized_mr_data, append_optimized_mr_to_master

# ─── Run the selected model ───────────────────────────────────────────────────

vlm_module  = model_cfg["vlm_module"]
llm_module  = model_cfg["llm_module"]

print(f"\n{'='*60}")
print(f"🤖 Starting benchmark: {model_name}  ({NUM_RUNS} runs)")
print(f"   VLM: {vlm_module}  |  LLM: {llm_module}")
print(f"{'='*60}")

# ── Load VLM (Agent 1) ─────────────────────────────────────────────────
load_vlm_fn   = _import(vlm_module, "load_model")
analyze_ui_fn = _import(vlm_module, "analyze_ui")
save_ui_fn    = _import(vlm_module, "save_ui_data")
warmup_vlm_fn = _import(vlm_module, "warmup") if hasattr(
    __import__(vlm_module, fromlist=[""]), "warmup"
) else None

print(f"\nLoading VLM ({model_name})...")
vision_model, processor = load_vlm_fn()

# ── Load LLM (Agents 2/3/4) ────────────────────────────────────────────
load_llm_fn   = _import(llm_module, "load_text_model")
warmup_llm_fn = _import(llm_module, "warmup") if hasattr(
    __import__(llm_module, fromlist=[""]), "warmup"
) else None

print(f"\nLoading LLM ({model_name})...")
text_model, text_tokenizer = load_llm_fn()

# ── Warmup pass (NOT timed, NOT tracked — this is setup cost) ──────────
print(f"\n🔥 Running warmup passes for {model_name}...")
if warmup_vlm_fn:
    warmup_vlm_fn(vision_model, processor)
if warmup_llm_fn:
    warmup_llm_fn(text_model, text_tokenizer)
print(f"✅ Warmup done — starting {NUM_RUNS} timed run(s)\n")

# ── Multiple timed runs ─────────────────────────────────────────────────
for run_idx in range(1, NUM_RUNS + 1):

    if run_idx > 1:
        print(f"\n⏳ Cooldown {COOLDOWN_SECONDS}s before run {run_idx}...")
        time.sleep(COOLDOWN_SECONDS)

    print(f"\n── Run {run_idx}/{NUM_RUNS} | Model: {model_name} ──")

    # Clear master CSVs fresh for each run (keyed by model+run in outputs)
    run_out_dir = f"outputs/{model_name}/run_{run_idx}"
    os.makedirs(run_out_dir, exist_ok=True)

    for image_file in image_files:
        full_path = os.path.join(IMAGES_DIR, image_file)
        screen_id = os.path.splitext(image_file)[0]
        topic     = id_to_topic.get(screen_id, "unknown")

        print(f"\n  ── {model_name} | run {run_idx} | {image_file} (topic: {topic}) ──")

        try:
            # ── Agent 1 — Perception ─────────────────────────────────
            tracker = EmissionsTracker(
                project_name=f"agent1_{model_name}_{run_idx}_{screen_id}",
                output_dir="outputs", log_level="error", save_to_file=False,
            )
            tracker.start()
            _t0 = time.time()
            ui_data          = analyze_ui_fn(full_path, vision_model, processor)
            ui_data["topic"] = topic
            save_ui_fn(ui_data, out_dir=f"{run_out_dir}/ui_analysis")
            _log_emissions(model_name, run_idx, screen_id, "Agent1_Perception", tracker, time.time() - _t0)
            print(f"  ✅ Agent 1 complete")

            # ── Agent 2 — Test Case Generation ───────────────────────
            tracker = EmissionsTracker(
                project_name=f"agent2_{model_name}_{run_idx}_{screen_id}",
                output_dir="outputs", log_level="error", save_to_file=False,
            )
            tracker.start()
            _t0 = time.time()
            tc_data              = generate_test_cases(ui_data, text_model, text_tokenizer)
            tc_data["screen_id"] = screen_id
            tc_data["topic"]     = topic
            _log_emissions(model_name, run_idx, screen_id, "Agent2_Generation", tracker, time.time() - _t0)
            print(f"  ✅ Agent 2 — {len(tc_data.get('test_cases', []))} test cases")

            save_test_cases(tc_data, out_dir=f"{run_out_dir}/testcases")
            append_to_master_csv(tc_data, master_path=f"{run_out_dir}/testcases_master.csv")

            # ── Agent 3 — Metamorphic Testing ────────────────────────
            tracker = EmissionsTracker(
                project_name=f"agent3_{model_name}_{run_idx}_{screen_id}",
                output_dir="outputs", log_level="error", save_to_file=False,
            )
            tracker.start()
            _t0 = time.time()
            mr_data = generate_metamorphic_relations(tc_data, text_model, text_tokenizer)
            _log_emissions(model_name, run_idx, screen_id, "Agent3_Metamorphic", tracker, time.time() - _t0)
            print(f"  ✅ Agent 3 — {len(mr_data.get('metamorphic_relations', []))} MRs")

            save_mr_data(mr_data, out_dir=f"{run_out_dir}/metamorphic_relations")
            append_mr_to_master_csv(mr_data, master_path=f"{run_out_dir}/metamorphic_relations_master.csv")

            # ── Agent 4 — Optimization ────────────────────────────────
            tracker = EmissionsTracker(
                project_name=f"agent4_{model_name}_{run_idx}_{screen_id}",
                output_dir="outputs", log_level="error", save_to_file=False,
            )
            tracker.start()
            _t0 = time.time()
            opt_data = optimize_metamorphic_relations(mr_data, text_model, text_tokenizer)
            _log_emissions(model_name, run_idx, screen_id, "Agent4_Optimization", tracker, time.time() - _t0)
            print(f"  ✅ Agent 4 — {len(opt_data.get('optimized_relations', []))} optimized MRs")

            save_optimized_mr_data(opt_data, out_dir=f"{run_out_dir}/optimized_relations")
            append_optimized_mr_to_master(opt_data, master_path=f"{run_out_dir}/optimized_relations_master.csv")

            print(f"  ✅ Pipeline complete for {screen_id}")

        except Exception as e:
            traceback.print_exc()
            print(f"  ❌ Failed on {image_file}: {e}")
            continue

print(f"\n✅ {model_name} benchmark finished ({NUM_RUNS} runs)")
print(f"🌱 Emissions log saved → {EMISSIONS_LOG}")
print(f"\nNext: run the other model from its own venv with:")
other = [m for m in MODEL_NAMES if m != model_name]
if other:
    print(f"   python benchmark_pipeline.py --model {other[0]}")
print(f"\nThen merge + summarize both with:")
print(f"   python summarize_emissions.py")