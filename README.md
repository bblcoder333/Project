# Project# Multi-Agent Mobile UI Testing Pipeline — Sustainability Benchmark

A 4-agent pipeline that processes ENRICO dataset screenshots through UI perception,
test case generation, metamorphic relation generation, and test optimization —
benchmarked across multiple small language models (SLMs) for energy/emissions
cost using CodeCarbon.

## Pipeline overview

| Agent | Task | Model role |
|---|---|---|
| Agent 1 | UI Perception — describes screen components, layout, accessibility | Vision-language model (VLM) |
| Agent 2 | Test Case Generation — functional/negative/accessibility test cases | Text LLM |
| Agent 3 | Metamorphic Relation Generation — 6-category taxonomy (INVARIANCE, MONOTONICITY, INPUT_TRANSFORMATION, VALIDATION_CONSISTENCY, INTERACTION_CONSISTENCY, ROBUSTNESS) | Text LLM + deterministic backfill code |
| Agent 4 | Optimization/Prioritization — keep/reduce/deprioritize decisions per MR | Text LLM + deterministic coverage-rule enforcement |

**Note on Agents 3 and 4:** both rely on deterministic post-processing code in
addition to the LLM prompt — Agent 3 has backfill functions that can generate
entire metamorphic relations in code if the LLM misses a category, and Agent 4
has a coverage-rule enforcement layer that can override the LLM's decisions.


## Models benchmarked

| Model pair | Vision model | Text model | Params (approx) |
|---|---|---|---|
| Qwen | Qwen2-VL-7B-Instruct | Qwen2.5-7B-Instruct | 7B / 7B |
| Phi-3.5 | Phi-3.5-vision-instruct | Phi-3.5-mini-instruct | 4.2B / 3.8B |
| InternVL2 | InternVL2-8B | internlm2_5-7b-chat | 8B / 7B (matched pair, same lab) |

Each model runs **5 timed benchmark runs** across all screenshots in `images/`,
with per-agent energy (kWh), emissions (kg CO2), and duration logged via
CodeCarbon.

## Repository structure

```
├── benchmark_pipeline.py          # Main benchmark runner — run once per model
├── summarize_emissions.py         # Aggregates results across all models into summary tables + chart
├── design_topics.csv              # Screen ID -> topic mapping (ENRICO metadata)
├── images/                        # Input screenshots
├── stages/
│   ├── ui_analysis.py             # Agent 1 — Qwen2-VL
│   ├── ui_analysis_phi.py         # Agent 1 — Phi-3.5-vision
│   ├── ui_analysis_internvl.py    # Agent 1 — InternVL2-8B
│   ├── test_generation.py         # Agent 2 (shared across all models)
│   ├── text_model_phi.py          # Agent 2/3/4 text model loader — Phi-3.5-mini
│   ├── text_model_internlm.py     # Agent 2/3/4 text model loader — internlm2_5-7b-chat
│   ├── metamorphic_testing.py     # Agent 3 (shared across all models)
│   └── optimization.py            # Agent 4 (shared across all models)
├── test_agent1_only.py            # Smoke test: Agent 1 only (Phi)
├── test_agent2_only.py            # Smoke test: Agent 1 + Agent 2 chained (Phi)
└── outputs/
    ├── <model_name>/run_1 .. run_5/   # Per-run outputs (test cases, MRs, optimized MRs)
    └── emissions_log_<model_name>.csv # Per-model emissions log
```

**Note:** Qwen's model loader is `stages/test_generation.py`'s `load_text_model()`
(the original pipeline) — it doubles as both the Agent 2 stage file and the
Qwen text-model loader, unlike Phi/InternVL which have a dedicated
`text_model_<name>.py` loader file.

## Why two virtual environments

Qwen requires a modern `transformers` version. Phi-3.5-vision, Phi-3.5-mini,
InternVL2-8B, and internlm2_5-7b-chat all rely on custom `trust_remote_code=True`
modeling code that was written against an older `transformers` internal API and
was never updated by the model authors — running them under a modern
`transformers` version raises cache-API or attribute errors (`DynamicCache` /
`all_tied_weights_keys` depending on the model). Pinning `transformers==4.46.1`
in a separate venv resolves this for all three of Phi-3.5-vision, Phi-3.5-mini,
and InternVL2-8B.

| Venv | Used for | Key package versions |
|---|---|---|
| `.venv` | Qwen | `transformers==5.8.1`, `torch==2.11.0+cu128` |
| `.venv-vision` | Phi-3.5, InternVL2 | `transformers==4.46.1`, `torch==2.11.0+cu128` |

## Setup

### 1. Qwen environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch transformers accelerate pillow qwen-vl-utils json_repair codecarbon matplotlib pandas
```

### 2. Phi-3.5 / InternVL2 environment
```powershell
python -m venv .venv-vision
.\.venv-vision\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.46.1"
pip install accelerate pillow json_repair codecarbon einops timm sentencepiece protobuf matplotlib pandas
```

## Running the benchmark

Each model must be run **separately, from its own venv**, since Qwen and the
other two models cannot coexist in the same Python environment (see above).

```powershell
# Qwen — from .venv
python benchmark_pipeline.py --model qwen

# Phi-3.5 — from .venv-vision
python benchmark_pipeline.py --model phi35

# InternVL2 — from .venv-vision
python benchmark_pipeline.py --model internvl
```

Each run automatically:
- Wipes and rebuilds `outputs/<model_name>/` and `outputs/emissions_log_<model_name>.csv`
  (safe to re-run without manual cleanup)
- Runs 5 full pipeline passes across every image in `images/`, with a 60-second
  cooldown between runs
- Note: all agents use greedy decoding (`do_sample=False`), so outputs are
  deterministic across runs — the 5 runs measure energy/timing stability, not
  output variance

## Summarizing results

Once all models you want to compare have been benchmarked:
```powershell
python summarize_emissions.py --by-screen
```
This auto-discovers every `outputs/emissions_log_*.csv` file, merges them, and
produces:
- `outputs/emissions_summary_by_agent.csv` — mean/std energy, emissions, duration per model x agent
- `outputs/emissions_summary_by_run.csv` — total per model x run (sanity check across runs)
- `outputs/emissions_summary_by_screen.csv` — per-screen breakdown (with `--by-screen`)
- `outputs/emissions_energy_by_agent.png` — grouped bar chart

