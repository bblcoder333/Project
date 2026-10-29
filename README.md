# Multi-Agent Mobile UI Testing Pipeline — Sustainability Benchmark

A 5-agent pipeline that processes ENRICO dataset screenshots through UI perception,
test case generation, metamorphic relation generation, test optimization, and
energy-aware suite reduction — benchmarked across multiple small language models
(SLMs) for energy/emissions cost using CodeCarbon.

## Pipeline overview

| Agent   | Task                                                                                                                                                                  | Model role                                          |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| Agent 1 | UI Perception — describes screen components, layout, accessibility                                                                                                  | Vision-language model (VLM)                          |
| Agent 2 | Test Case Generation — functional/negative/accessibility test cases                                                                                                 | Text LLM                                             |
| Agent 3 | Metamorphic Relation Generation — 6-category taxonomy (INVARIANCE, MONOTONICITY, INPUT_TRANSFORMATION, VALIDATION_CONSISTENCY, INTERACTION_CONSISTENCY, ROBUSTNESS) | Text LLM + deterministic backfill code               |
| Agent 4 | Optimization/Prioritization — keep/reduce/deprioritize decision per MR                                                                                              | Text LLM + deterministic coverage-rule enforcement   |
| Agent 5 | Suite Reduction & Energy Savings Estimator — assigns each optimized MR an execution tier (always/sampled/optional/dropped) + cycle frequency                       | Text LLM, constrained by a hard-coded allowed-tier set per Agent 4 decision, with a deterministic fallback mapping if the LLM output fails to parse |

**Note on Agents 3, 4, and 5:** each layers deterministic post-processing on top of
the LLM call. Agent 3 has backfill functions that generate an entire metamorphic
relation in code if the LLM misses a category. Agent 4 has a coverage-rule
enforcement layer that can override the LLM's decision. Agent 5's energy/percentage
math is always computed in plain Python from the assigned tier — never by the LLM
— and falls back to a fixed `DECISION_TO_TIER_FALLBACK` mapping if its own LLM call
can't be parsed.

**Agent ↔ phase naming.** The paper and result workbook call these *phases*:
Phase 1 = Agent 1 (UI Perception), Phase 2 = Agent 2 (Source Test Generation),
Phase 3 = Agent 3 (MR Generation), Phase 4 = Agent 4 (MR Prioritization),
Phase 5 = Agent 5 (Execution Scheduling). The code and logs use "Agent N".

**Robustness layers added to the LLM stages** (all deterministic, no extra LLM calls):

- **Agents 2 & 3 — type sanitizing.** `_sanitize_test_cases` / `_sanitize_tc_data`
  coerce non-string elements (nested lists/dicts some models emit inside `steps`,
  `expected`, `test_data`) to strings before any `.join()` / `re.sub()`. Without
  this, one malformed test case raised an exception that discarded *every* test
  case for the screen.
- **Agent 4 — hallucination filter and truncation backfill.**
  `_filter_hallucinated_mrs` drops `mr_id`s the LLM invented that Agent 3 never
  produced; `_backfill_missing_mrs` re-adds any Agent 3 MR missing from the response
  (e.g. cut off at the output-token limit), labeled `Backfilled` in `reason`.

## Two ways to run this

`batch_runner.py` (the wrapper that looped `main.py` in batches) is gone — its
job is now built into `benchmark_pipeline.py` itself. `main.py` stays as a
simple, single-model backup option:

| | `main.py` | `benchmark_pipeline.py` |
| --- | --- | --- |
| Purpose | Simple, single-model backup — process the full dataset once, with Qwen | Compare energy/emissions across models |
| Models | Qwen only (hard-coded) | `qwen`, `phi35`, `internvl`, `internvl_mistral`, `internvl_llama3` (see `MODELS` list) |
| Resume support | Yes — checkpoints to `outputs/completed_screens.txt` | Yes — checkpoints to `outputs/<model>/run_N/completed_screens.txt`, per run |
| Batching | `--batch-size N` processes the next N unfinished images (no default — omit to process all remaining in one go) | `--batch-size N` per run, defaults to 100 for every model, plus `--max-batches` to cap an invocation |
| Runs | Always a single pass | `--runs` timed repeats (default 3) per model |
| Output layout | Flat `outputs/` | Namespaced `outputs/<model_name>/run_N/` |
| Warmup stabilization | No | Yes — 300s wait after warmup before the first timed run, so leftover GPU/driver activity doesn't skew Run 1 |
| `--fresh` | Wipes `outputs/` entirely | Wipes `outputs/<model_name>/` and that model's emissions log only |

Use `main.py` when you just want one clean, resumable Qwen pass without the
multi-model machinery. Use `benchmark_pipeline.py` when you're comparing
models or want the finer-grained per-run/per-batch output structure.

```powershell
# benchmark_pipeline.py — all models default to 100-image batches per run
python benchmark_pipeline.py --model qwen
python benchmark_pipeline.py --model phi35 --batch-size 250
python benchmark_pipeline.py --model internvl --max-batches 3   # quick test, still batches of 100
python benchmark_pipeline.py --model internvl_mistral
python benchmark_pipeline.py --model internvl_llama3
python benchmark_pipeline.py --model qwen --runs 1
python benchmark_pipeline.py --model qwen --fresh
```

⚠️ **`--fresh` on `benchmark_pipeline.py` deletes `outputs/<model_name>/`
entirely, including its checkpoints and batch logs**, plus that model's
emissions log. **`--fresh` on `main.py` deletes all of `outputs/`, including
its checkpoint file.** Only pass either if you actually want to discard
progress and start over.

## Models benchmarked

Five configurations, each a vision model (Agent 1) plus a text LLM shared by
Agents 2–5. All weights are loaded in `bfloat16` (no quantization).

| `--model`          | Vision model (Agent 1)  | Text LLM (Agents 2–5)                | Params (approx)  | Venv           |
| ------------------ | ----------------------- | ------------------------------------ | ---------------- | -------------- |
| `qwen`             | Qwen2-VL-7B-Instruct    | Qwen2.5-7B-Instruct                  | 7B / 7B          | `.venv`        |
| `phi35`            | Phi-3.5-vision-instruct | Phi-3.5-mini-instruct                | 4.2B / 3.8B      | `.venv-vision` |
| `internvl`         | InternVL2-8B            | internlm2_5-7b-chat                  | 8B / 7B          | `.venv-vision` |
| `internvl_mistral` | InternVL2-8B            | Mistral-7B-Instruct-v0.3             | 8B / 7B          | `.venv-vision` |
| `internvl_llama3`  | InternVL2-8B            | Meta-Llama-3.1-8B-Instruct           | 8B / 8B          | `.venv-vision` |

All five are wired into `benchmark_pipeline.py`'s `MODELS` list. Each does `--runs`
timed passes (default 3) across every screenshot in `images/`, with per-agent
energy (kWh), emissions (kg CO2), and duration logged via CodeCarbon.

## Generation settings

Every call uses greedy decoding (`do_sample=False`). Agents 2–5 additionally set
`top_p=1.0` and `repetition_penalty=1.05`. The stage files are shared, so these
limits are identical for every text LLM.

| Agent | Prompt truncation (`max_length`) | `max_new_tokens` | Notes |
| ----- | -------------------------------- | ---------------- | ----- |
| 1 — UI Perception | model-specific image preprocessing | 4096 | Qwen2-VL: `min/max_pixels` = 256/1280 × 28²; InternVL2: 448px tiles, up to 12 + thumbnail; Phi-3.5-vision: `num_crops=16` |
| 2 — Test Case Generation | 6144 | 4000 | |
| 3 — MR Generation | 6144 | 6000 | |
| 4 — MR Prioritization | 8192 | 4000 | |
| 5 — Suite Reduction | 4096 | 2000 | |

**Token-limit history (reproducibility).** Commit `fbc1a2e` (2026-10-29) raised
Agent 2's prompt limit from 4096 to 6144, and Agent 4's prompt limit from 6144 to
8192 and `max_new_tokens` from 2000 to 4000. Qwen and Phi-3.5 results were
collected before this change and have not been re-run with the new limits; the
InternVL + Mistral / Llama-3.1 configurations were run with them. Check out an
earlier commit to reproduce the older settings.

**Measurement.** One CodeCarbon `EmissionsTracker` per agent per screen
(`save_to_file=False`; values are appended to `outputs/emissions_log_<model>.csv`
along with the detected region/country). Before timed runs: model warmup, then a
300 s stabilization wait; 60 s cooldown between runs.

## Repository structure

```
├── main.py                        # Simple single-model backup — full-dataset run with Qwen,
│                                   #   resumable via checkpoint + --batch-size, flat outputs/
├── benchmark_pipeline.py          # Multi-model comparison runner — resumable, batched, per-model outputs
├── requirements.txt               # Qwen env (.venv); requirements-vision.txt is for .venv-vision
├── breakeven_check.py             # Reads outputs from a main.py run; computes how many regression
│                                   #   cycles it takes for Agent 5's own generation cost to pay for
│                                   #   itself via its projected per-cycle savings
├── summarize_emissions.py         # Aggregates outputs/emissions_log_*.csv (from benchmark_pipeline.py)
│                                   #   into summary tables + chart
├── design_topics.csv              # Screen ID -> topic mapping (ENRICO metadata) — tracked in git
├── images/                        # Input screenshots (ENRICO) — tracked in git, ready to use after clone
├── stages/
│   ├── ui_analysis.py             # Agent 1 — Qwen2-VL
│   ├── ui_analysis_phi.py         # Agent 1 — Phi-3.5-vision
│   ├── ui_analysis_internvl.py    # Agent 1 — InternVL2-8B (shared by internvl / internvl_mistral / internvl_llama3)
│   ├── test_generation.py         # Agent 2 (shared) — also Qwen's load_text_model()
│   ├── text_model_phi.py          # Agent 2/3/4/5 text model loader — Phi-3.5-mini
│   ├── text_model_internlm.py     # Agent 2/3/4/5 text model loader — internlm2_5-7b-chat
│   ├── text_model_mistral.py      # Agent 2/3/4/5 text model loader — Mistral-7B-Instruct-v0.3
│   ├── text_model_llama3.py       # Agent 2/3/4/5 text model loader — Meta-Llama-3.1-8B-Instruct
│   ├── metamorphic_testing.py     # Agent 3 (shared)
│   ├── optimization.py            # Agent 4 (shared)
│   └── optimization_reduction.py  # Agent 5 (shared)
├── test_agent1.py .. test_agent4.py  # Smoke tests, one agent (or agent chain) deeper each time
├── test_agent5.py                 # Smoke test: Agent 5 over outputs/optimized_relations/, any --model
└── outputs/                       # Not tracked in git — created by main.py / benchmark_pipeline.py
    ├── completed_screens.txt      # main.py's checkpoint file
    ├── emissions_log.csv          # main.py's single-model emissions log
    ├── emissions_log_<model>.csv  # benchmark_pipeline.py's per-model emissions log
    └── <model_name>/
        └── run_1 .. run_N/        # benchmark_pipeline.py per-run outputs
            ├── completed_screens.txt   # Checkpoint — which screens finished all 5 agents
            ├── batch_log.csv           # One row per batch: timing, images processed, status
            ├── next_batch_id.txt       # Persistent batch-id counter (survives Ctrl+C/restart)
            └── testcases_master.csv, metamorphic_relations_master.csv, etc.
```

**Note:** `stages/test_generation.py`'s `load_text_model()` doubles as both the
Agent 2 stage file and Qwen's text-model loader — Phi, InternLM, Mistral and Llama-3.1
have their own dedicated `text_model_<name>.py` loader instead.

## Why two virtual environments

Qwen requires a modern `transformers` version. Phi-3.5-vision, Phi-3.5-mini,
InternVL2-8B, and internlm2_5-7b-chat all rely on custom `trust_remote_code=True`
modeling code written against an older `transformers` internal API that was never
updated by the model authors — running them under a modern `transformers` version
raises cache-API or attribute errors (`DynamicCache` / `all_tied_weights_keys`
depending on the model). Pinning `transformers==4.46.1` in a separate venv resolves
this for Phi-3.5-vision, Phi-3.5-mini, and InternVL2-8B. Mistral and Llama-3.1 load
natively (no `trust_remote_code`) and run fine under 4.46.1, so the InternVL2 + Mistral
and InternVL2 + Llama-3.1 pairings use `.venv-vision` too.

| Venv           | Used for            | Key package versions                          |
| -------------- | -------------------- | --------------------------------------------- |
| `.venv`        | Qwen                  | `transformers==5.8.1`, `torch==2.11.0+cu128`  |
| `.venv-vision` | Phi-3.5, all InternVL2 pairings (InternLM, Mistral, Llama-3.1) | `transformers==4.46.1`, `torch==2.11.0+cu128` |

## Setup

**Requirements:** Python 3.10+ (matching what this was developed/tested against),
and a CUDA-capable GPU — each model pair (VLM + text LLM, ~7-8B params each) needs
roughly 24GB+ VRAM loaded together in bf16. Running on less will likely OOM.

**Hugging Face access:** all models are pulled from the Hugging Face Hub on first
run. Mistral-7B-Instruct-v0.3 and Meta-Llama-3.1-8B-Instruct are **gated**: accept
the terms on each model page (Llama approval is manual and may not be instant). If any
other model requires accepting a license, run `huggingface-cli login` (or set `HF_TOKEN`) first — otherwise
the first `from_pretrained()` call will fail with a 401. Internet access is
required at least for this first run per model, to download weights into the
local HF cache; after that, it may be able to run offline (untested here).

### 1. Qwen environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 2. Phi-3.5 / InternVL2 environment

```powershell
python -m venv .venv-vision
.\.venv-vision\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-vision.txt
```

⚠️ Before running `--model internvl` for real, do a quick standalone check that
`internlm2_5-7b-chat`'s tokenizer has a working `chat_template` — see the
compatibility note at the top of `stages/text_model_internlm.py`. If it's
missing, `warmup()` will fail loudly (by design) rather than silently producing
garbage output, but better to catch it before a long run.

`images/` and `design_topics.csv` are both tracked in git, so they come straight
from `git clone` — no setup needed for those.

## Running the full dataset (simple backup: `main.py`, Qwen)

```powershell
# process the next 1000 not-yet-completed images
python main.py --batch-size 1000

# process everything still remaining
python main.py

# start completely over — wipes outputs/, including the checkpoint (use with care)
python main.py --fresh
```

Each run prints how many screens are already done, how many remain, and picks up
exactly where the last run left off via `outputs/completed_screens.txt`. A screen
is only marked complete after **all 5 agents** succeed on it — a failure mid-screen
leaves it unmarked so it's retried on the next run.

After a run, check the generation-vs-savings tradeoff:

```powershell
python breakeven_check.py
```

## Comparing models (`benchmark_pipeline.py`)

Each model must be run **separately, from its own venv** — Qwen and the other
models can't coexist in the same Python environment (see above).

```powershell
# Qwen — from .venv
python benchmark_pipeline.py --model qwen

# Phi-3.5 — from .venv-vision
python benchmark_pipeline.py --model phi35

# InternVL2 pairings — from .venv-vision
python benchmark_pipeline.py --model internvl
python benchmark_pipeline.py --model internvl_mistral
python benchmark_pipeline.py --model internvl_llama3
```

By default this does `--runs` (3) timed passes over every image in `images/`,
each run processed in batches of 100 images (checkpointed after every batch),
with a 60-second cooldown between runs and a 300-second stabilization wait
after warmup before Run 1 starts. All agents use greedy decoding
(`do_sample=False`), so the runs measure energy/timing stability, not output
variance.

To use a different batch size:

```powershell
python benchmark_pipeline.py --model qwen --batch-size 200
```

Or to go back to the old "everything in one batch" behavior:

```powershell
python benchmark_pipeline.py --model qwen --batch-size 0
```

Stop it (Ctrl+C) and re-run the exact same command any time — it picks up from
`outputs/qwen/run_N/completed_screens.txt` rather than starting over.

For a quick smoke test without committing to a full run:

```powershell
python benchmark_pipeline.py --model qwen --batch-size 10 --max-batches 1
```

### Smoke-testing a single agent (any model)

```powershell
python test_agent1.py --model qwen        # Agent 1 only
python test_agent2.py --model phi35       # through Agent 2
python test_agent3.py --model internvl    # through Agent 3
python test_agent4.py --model qwen        # through Agent 4
python test_agent5.py --model qwen        # Agent 5 over an existing outputs/optimized_relations/ run
```

## Summarizing results across models

Once you've benchmarked the models you want to compare:

```powershell
python summarize_emissions.py --by-screen
```

Auto-discovers every `outputs/emissions_log_*.csv`, merges them, and produces:

- `outputs/emissions_summary_by_agent.csv` — mean/std energy, emissions, duration per model x agent
- `outputs/emissions_summary_by_run.csv` — total per model x run (sanity check across runs)
- `outputs/emissions_summary_by_screen.csv` — per-screen breakdown (with `--by-screen`)
- `outputs/emissions_energy_by_agent.png` — grouped bar chart

Flags: `--log <path>` (repeatable) to point at specific logs, `--log-dir` to change
where it searches (default `outputs/`), `--out` to change where summaries are written.