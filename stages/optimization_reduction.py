"""
stages/optimization_reduction.py
Agent 5 — Suite Reduction & Energy Savings Estimator (LLM-based)

Takes Agent 4's optimized_relations (each MR labeled with a decision:
keep / high_priority_keep / reduce_repetitions / partial_sampling /
lower_priority) and assigns each MR a concrete execution tier
(always / sampled / optional / dropped) plus a cycle_frequency, via an LLM
call constrained by a hard-coded allowed-tier set per decision. Falls back
to a fixed deterministic mapping if the LLM output fails to parse.

Energy/percentage math is computed in plain Python from the assigned
tier/frequency, never by the LLM.

Uses the SAME shared text model + tokenizer already loaded for Agents 2/3/4.
"""

import csv
import json
import os
import re

from json_repair import repair_json

# ─── Fallback tier mapping (used only if the LLM call/parse fails) ───────────

DECISION_TO_TIER_FALLBACK = {
    "high_priority_keep": "always",
    "keep":                "always",
    "partial_sampling":    "sampled",
    "lower_priority":      "optional",
    "reduce_repetitions":  "dropped",
}

# ─── Hard constraints: which tiers each Agent-4 decision is ALLOWED to map to
# The LLM chooses within this set; anything else gets clamped back to the
# fallback default for that decision. This is what keeps Agent 5 safe even
# though it is now a judgment call rather than a fixed lookup.

ALLOWED_TIERS_BY_DECISION = {
    "high_priority_keep": {"always"},
    "keep":                {"always", "sampled"},
    "partial_sampling":    {"sampled", "optional"},
    "lower_priority":      {"optional", "sampled"},
    "reduce_repetitions":  {"dropped", "optional"},
}

# cycle_frequency ranges per tier — clamps out-of-range LLM picks rather
# than rejecting the whole MR's tier choice.
FREQUENCY_RANGE_BY_TIER = {
    "always":  (1, 1),
    "sampled": (2, 10),
    "optional": (5, 30),
    "dropped": (0, 0),
}

# Documented assumption: energy cost of executing ONE metamorphic relation
# once (device/emulator UI interaction + assertion). NOT measured — a stated
# placeholder. Adjust if you have a better estimate and cite it in your report.
DEFAULT_PER_TEST_EXECUTION_KWH = 0.0005

REDUCED_CSV_COLUMNS = [
    "mr_id", "source_tc_id", "screen_id", "topic", "mr_category",
    "transformation", "decision", "execution_tier", "cycle_frequency", "reasoning",
]

SUMMARY_CSV_COLUMNS = [
    "screen_id", "topic", "total_mrs", "always_count", "sampled_count",
    "optional_count", "dropped_count", "active_count", "reduction_pct",
    "cost_per_cycle_unoptimized_kwh", "cost_per_cycle_optimized_kwh",
    "energy_savings_pct", "llm_tier_count", "clamped_tier_count",
    "regenerated_reasoning_count", "fallback_tier_count",
]

# ─── Prompt ───────────────────────────────────────────────────────────────────

AGENT5_PROMPT_TEMPLATE = """
You are Agent 5, a Sustainability-Oriented Test Execution Scheduling Agent.

Agent 4 already labeled every metamorphic relation (MR) with a decision
(keep / high_priority_keep / reduce_repetitions / partial_sampling /
lower_priority). Your job is DIFFERENT from Agent 4's: decide how often each
MR should actually EXECUTE in a real, repeating regression suite, to
minimize wasted device/emulator execution energy while preserving real bug-
catching value.

---

## INPUT
Screen ID: {screen_id}
Topic: {topic}

OPTIMIZED MRs FROM AGENT 4 (columns separated by ::):
{mr_csv}

---

## EXECUTION TIERS (use exactly these strings)
- "always"   -> runs every regression cycle
- "sampled"  -> runs every Nth cycle (choose N)
- "optional" -> runs only in periodic/nightly full regression (choose N)
- "dropped"  -> never re-run (true redundant repeat of another MR)

## HARD RULE — ALLOWED TIERS PER DECISION (you MUST stay within these)
- decision = "high_priority_keep" -> tier MUST be "always". No exceptions.
- decision = "keep" -> tier MUST be "always" OR "sampled".
- decision = "partial_sampling" -> tier MUST be "sampled" OR "optional".
- decision = "lower_priority" -> tier MUST be "optional" OR "sampled".
- decision = "reduce_repetitions" -> tier MUST be "dropped" OR "optional".
Never pick a tier outside the allowed set for that MR's decision.

## HOW TO CHOOSE WITHIN THE ALLOWED SET (this is the actual judgment call)
1. CORRECTNESS-CRITICAL categories (MONOTONICITY, VALIDATION_CONSISTENCY,
   INPUT_TRANSFORMATION) that are "keep" -> stay "always". These catch real
   computation/validation regressions; sampling them risks missing bugs.
2. STABILITY-CHECK categories (ROBUSTNESS, INTERACTION_CONSISTENCY) that are
   "keep" -> may be downgraded to "sampled" (choose cycle_frequency 2-4).
   These re-check "does the app still not crash", which has diminishing
   value once it has passed several consecutive cycles.
3. For "partial_sampling" MRs: pick cycle_frequency 2-5 for MRs with higher
   fault-detection potential (boundary values, unit conversions), 5-10 for
   lower-value repeats.
4. For "lower_priority" MRs: pick cycle_frequency 10-30 depending on how
   marginal the check is; INVARIANCE tab-switch checks on simple static
   content can go toward the higher end (20-30).
5. For "reduce_repetitions" MRs: default to "dropped" (true duplicate of an
   already-covered MR). Only use "optional" instead of "dropped" if the
   reason text explicitly suggests it still has SOME distinct value (e.g.
   different icon target) even though it is a repeat pattern.

## OUTPUT
Return ONLY valid JSON, no markdown, no explanation.
The example below shows FORMAT ONLY. Your "reasoning" must describe the
ACTUAL mr_category and transformation of the MR you are assigning — never
reuse the example's wording, since it is not about any real MR here.

{{
  "screen_id": "{screen_id}",
  "tier_assignments": [
    {{
      "mr_id": "MR-01",
      "execution_tier": "always",
      "cycle_frequency": 1,
      "reasoning": "<one sentence specific to this MR's own category and transformation>"
    }}
  ]
}}

Every mr_id from the input MUST appear exactly once in tier_assignments.
"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _opt_data_to_csv_string(opt_data: dict) -> str:
    lines = ["mr_id :: source_tc_id :: mr_category :: transformation :: decision :: reason"]
    for mr in opt_data.get("optimized_relations", []):
        lines.append(
            f"{mr.get('mr_id','')} :: {mr.get('source_tc_id','')} :: "
            f"{mr.get('mr_category','')} :: {mr.get('transformation','')} :: "
            f"{mr.get('decision','')} :: {mr.get('reason','')}"
        )
    return "\n".join(lines)


def _build_mr_lookup(opt_data: dict) -> dict:
    return {mr.get("mr_id", ""): mr for mr in opt_data.get("optimized_relations", [])}


def _clamp_frequency(tier: str, freq) -> int:
    lo, hi = FREQUENCY_RANGE_BY_TIER.get(tier, (1, 1))
    try:
        freq = int(freq)
    except (TypeError, ValueError):
        freq = lo
    return max(lo, min(hi, freq))


def _fallback_tier_for(decision: str) -> tuple:
    """Returns (tier, cycle_frequency, reasoning) using the safe fixed mapping."""
    tier = DECISION_TO_TIER_FALLBACK.get(decision, "always")
    default_freq = {"always": 1, "sampled": 3, "optional": 10, "dropped": 0}[tier]
    reasoning = f"Fallback mapping — decision '{decision}' defaults to '{tier}'"
    return tier, default_freq, reasoning


# Phrases known to appear in copied/unfilled LLM output rather than genuine
# per-MR reasoning. Checked against each MR's own category/transformation to
# catch cases where the model reused wording that doesn't apply to it.
_KNOWN_TEMPLATE_PHRASES = ["waist ratio computation", "total computation"]


def _is_untrustworthy_reasoning(reasoning: str, mr: dict) -> bool:
    """True if reasoning looks copied from a template/example rather than
    written for this specific MR."""
    text = reasoning.lower()

    if "<" in reasoning and ">" in reasoning:
        return True  # literal unfilled placeholder

    grounding = (mr.get("mr_category", "") + " " + mr.get("transformation", "")).lower()
    for phrase in _KNOWN_TEMPLATE_PHRASES:
        keyword = phrase.split()[0]  # e.g. "waist" or "total"
        if phrase in text and keyword not in grounding:
            return True

    return False


def _generic_reasoning(tier: str, mr: dict) -> str:
    category = mr.get("mr_category", "MR")
    return f"Assigned '{tier}' tier for {category} MR based on its Agent 4 decision"


def _validate_and_clean_assignments(assignments: list, mr_lookup: dict) -> tuple:
    """
    Enforces the hard per-decision tier constraints and frequency ranges.
    Returns (cleaned_assignments_by_mr_id, llm_used_count, clamped_count,
    regenerated_count). Any assignment referencing an unknown mr_id is
    dropped; any mr_id from Agent 4 missing from the LLM output gets the
    fallback applied later.
    """
    cleaned = {}
    llm_used_count = 0
    clamped_count = 0
    regenerated_count = 0

    for a in assignments:
        mr_id = a.get("mr_id", "")
        if mr_id not in mr_lookup:
            continue  # hallucinated mr_id — ignore

        mr = mr_lookup[mr_id]
        decision = mr.get("decision", "keep")
        allowed = ALLOWED_TIERS_BY_DECISION.get(decision, {"always"})

        tier = a.get("execution_tier", "")
        if tier not in allowed:
            # LLM picked a tier outside what this decision permits — clamp
            # to the safe fallback rather than trust the LLM's pick here.
            tier, freq, reasoning = _fallback_tier_for(decision)
            reasoning = (
                f"LLM tier '{a.get('execution_tier','')}' not allowed for decision "
                f"'{decision}' — clamped to safe default '{tier}'"
            )
            clamped_count += 1
        else:
            freq = _clamp_frequency(tier, a.get("cycle_frequency", 1))
            reasoning = (a.get("reasoning") or "").strip()
            if not reasoning or _is_untrustworthy_reasoning(reasoning, mr):
                reasoning = _generic_reasoning(tier, mr)
                regenerated_count += 1
            llm_used_count += 1

        cleaned[mr_id] = {
            "execution_tier": tier,
            "cycle_frequency": freq,
            "reasoning": reasoning,
        }

    return cleaned, llm_used_count, clamped_count, regenerated_count


def _fill_missing_with_fallback(cleaned: dict, mr_lookup: dict) -> int:
    """Any MR the LLM never assigned gets the deterministic fallback. Returns
    how many MRs needed the fallback (for diagnostics/reporting)."""
    fallback_count = 0
    for mr_id, mr in mr_lookup.items():
        if mr_id in cleaned:
            continue
        decision = mr.get("decision", "keep")
        tier, freq, reasoning = _fallback_tier_for(decision)
        cleaned[mr_id] = {
            "execution_tier": tier,
            "cycle_frequency": freq,
            "reasoning": reasoning + " (LLM did not return an assignment for this MR)",
        }
        fallback_count += 1
    return fallback_count


# ─── Main LLM-based tiering function ──────────────────────────────────────────

def _assign_tiers_via_llm(opt_data: dict, model, tokenizer) -> tuple:
    """
    Calls the shared text LLM to assign execution tiers + cycle_frequency
    per MR. Returns (assignments_dict, llm_used_count, clamped_count,
    regenerated_count, fallback_count).
    On total parse failure, returns an all-fallback assignment (llm_used=0).
    """
    import torch

    screen_id = opt_data.get("screen_id", "unknown")
    topic = opt_data.get("topic", "unknown")
    mr_lookup = _build_mr_lookup(opt_data)

    if not mr_lookup:
        return {}, 0, 0, 0

    mr_csv = _opt_data_to_csv_string(opt_data)
    prompt = AGENT5_PROMPT_TEMPLATE.format(screen_id=screen_id, topic=topic, mr_csv=mr_csv)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a test execution scheduling expert focused on minimizing "
                "energy use. Output ONLY valid JSON with no markdown or explanation."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to("cuda")

    print("   🔄 Assigning execution tiers (Agent 5, LLM-based)...")

    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=2000,
            do_sample=False,
            top_p=1.0,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    try:
        blocks = re.findall(r"\{.*\}", raw, re.DOTALL)
        if not blocks:
            print("   ⚠️  Agent 5: No JSON block found — using fallback tiers for all MRs")
            cleaned = {}
        else:
            candidate = max(blocks, key=len)
            repaired = repair_json(candidate)
            result = json.loads(repaired)
            assignments = result.get("tier_assignments", [])
            cleaned, llm_used, clamped, regenerated = _validate_and_clean_assignments(assignments, mr_lookup)
            fallback_count = _fill_missing_with_fallback(cleaned, mr_lookup)
            print(f"   ✅ Agent 5 — {llm_used} tiers from LLM, {clamped} clamped, "
                  f"{regenerated} reasoning regenerated, {fallback_count} fallback")
            return cleaned, llm_used, clamped, regenerated, fallback_count

    except Exception as e:
        print(f"   ❌ Agent 5 JSON parse failed: {e} — using fallback tiers for all MRs")

    cleaned = {}
    fallback_count = _fill_missing_with_fallback(cleaned, mr_lookup)
    return cleaned, 0, 0, 0, fallback_count


# ─── Deterministic math — never trust the LLM to do this part ───────────────

def reduce_and_tier(opt_data: dict, model, tokenizer) -> dict:
    """
    Agent 5 main function. Calls the LLM to assign execution tiers per MR
    (with hard safety-net validation), then deterministically builds:
      - reduced_relations: every MR tagged with its execution_tier,
        cycle_frequency, and reasoning
      - summary: counts per tier, active suite size, and diagnostics on how
        many tiers came from the LLM vs. the safety-net fallback
    """
    screen_id = opt_data.get("screen_id", "unknown")
    topic = opt_data.get("topic", "unknown")
    relations = opt_data.get("optimized_relations", [])

    assignments, llm_used_count, clamped_count, regenerated_count, fallback_count = _assign_tiers_via_llm(opt_data, model, tokenizer)

    reduced = []
    tier_counts = {"always": 0, "sampled": 0, "optional": 0, "dropped": 0}

    for mr in relations:
        mr_id = mr.get("mr_id", "")
        a = assignments.get(mr_id)
        if a is None:
            tier, freq, reasoning = _fallback_tier_for(mr.get("decision", "keep"))
        else:
            tier, freq, reasoning = a["execution_tier"], a["cycle_frequency"], a["reasoning"]

        tier_counts[tier] += 1
        reduced.append({
            "mr_id": mr_id,
            "source_tc_id": mr.get("source_tc_id", ""),
            "mr_category": mr.get("mr_category", ""),
            "transformation": mr.get("transformation", ""),
            "decision": mr.get("decision", ""),
            "execution_tier": tier,
            "cycle_frequency": freq,
            "reasoning": reasoning,
        })

    total = len(relations)
    active = tier_counts["always"] + tier_counts["sampled"] + tier_counts["optional"]
    reduction_pct = round((1 - active / total) * 100, 2) if total else 0.0

    summary = {
        "screen_id": screen_id,
        "topic": topic,
        "total_mrs": total,
        "always_count": tier_counts["always"],
        "sampled_count": tier_counts["sampled"],
        "optional_count": tier_counts["optional"],
        "dropped_count": tier_counts["dropped"],
        "active_count": active,
        "reduction_pct": reduction_pct,
        "llm_tier_count": llm_used_count,
        "clamped_tier_count": clamped_count,
        "regenerated_reasoning_count": regenerated_count,
        "fallback_tier_count": fallback_count,
    }

    return {
        "screen_id": screen_id,
        "topic": topic,
        "reduced_relations": reduced,
        "summary": summary,
    }


def estimate_energy_savings(reduced_data: dict,
                             per_test_execution_kwh: float = DEFAULT_PER_TEST_EXECUTION_KWH) -> dict:
    """
    Pure arithmetic — NEVER delegated to the LLM. Computes per-regression-
    cycle execution energy, unoptimized vs. optimized, using each MR's
    actual cycle_frequency (not a flat constant), so MRs the LLM sampled
    at different rates are amortized individually rather than bucketed.
    """
    summary = reduced_data["summary"]
    relations = reduced_data["reduced_relations"]

    unoptimized_cost = summary["total_mrs"] * per_test_execution_kwh

    optimized_cost = 0.0
    for mr in relations:
        freq = mr.get("cycle_frequency", 0)
        if freq > 0:
            optimized_cost += per_test_execution_kwh / freq
        # freq == 0 (dropped) contributes nothing

    savings_pct = (
        round((1 - optimized_cost / unoptimized_cost) * 100, 2)
        if unoptimized_cost > 0 else 0.0
    )

    summary["cost_per_cycle_unoptimized_kwh"] = round(unoptimized_cost, 8)
    summary["cost_per_cycle_optimized_kwh"] = round(optimized_cost, 8)
    summary["energy_savings_pct"] = savings_pct

    return reduced_data


def generate_reduced_suite(opt_data: dict, model, tokenizer,
                            per_test_execution_kwh: float = DEFAULT_PER_TEST_EXECUTION_KWH) -> dict:
    """Convenience wrapper: reduce_and_tier() + estimate_energy_savings()."""
    reduced_data = reduce_and_tier(opt_data, model, tokenizer)
    reduced_data = estimate_energy_savings(reduced_data, per_test_execution_kwh)

    s = reduced_data["summary"]
    print(f"   ✅ Agent 5 reduced {s['total_mrs']} MRs -> {s['active_count']} active "
          f"({s['reduction_pct']}% suite reduction, ~{s['energy_savings_pct']}% "
          f"projected per-cycle execution energy savings; "
          f"{s['llm_tier_count']} LLM tiers, {s['clamped_tier_count']} clamped, "
          f"{s['regenerated_reasoning_count']} reasoning regenerated, "
          f"{s['fallback_tier_count']} fallback)")

    return reduced_data


# ─── Save functions ─────────────────────────────────────────────────────────────

def save_reduced_suite(reduced_data: dict, out_dir: str = "outputs/reduced_suite"):
    os.makedirs(out_dir, exist_ok=True)
    screen_id = reduced_data.get("screen_id", "unknown")
    topic = reduced_data.get("topic", "unknown")
    relations = reduced_data.get("reduced_relations", [])

    out_path = os.path.join(out_dir, f"{screen_id}_Reduced.csv")
    if os.path.exists(out_path):
        os.remove(out_path)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REDUCED_CSV_COLUMNS)
        w.writeheader()
        for mr in relations:
            w.writerow({
                "mr_id": mr.get("mr_id", ""),
                "source_tc_id": mr.get("source_tc_id", ""),
                "screen_id": screen_id,
                "topic": topic,
                "mr_category": mr.get("mr_category", ""),
                "transformation": mr.get("transformation", ""),
                "decision": mr.get("decision", ""),
                "execution_tier": mr.get("execution_tier", ""),
                "cycle_frequency": mr.get("cycle_frequency", ""),
                "reasoning": mr.get("reasoning", ""),
            })

    print(f"  💾 Reduced suite saved → {out_path}")
    return out_path


def append_reduced_to_master(reduced_data: dict,
                              master_path: str = "outputs/reduced_suite_master.csv"):
    os.makedirs(os.path.dirname(master_path) or ".", exist_ok=True)
    screen_id = reduced_data.get("screen_id", "unknown")
    topic = reduced_data.get("topic", "unknown")
    relations = reduced_data.get("reduced_relations", [])

    exists = os.path.exists(master_path)
    with open(master_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REDUCED_CSV_COLUMNS)
        if not exists:
            w.writeheader()
        for mr in relations:
            w.writerow({
                "mr_id": mr.get("mr_id", ""),
                "source_tc_id": mr.get("source_tc_id", ""),
                "screen_id": screen_id,
                "topic": topic,
                "mr_category": mr.get("mr_category", ""),
                "transformation": mr.get("transformation", ""),
                "decision": mr.get("decision", ""),
                "execution_tier": mr.get("execution_tier", ""),
                "cycle_frequency": mr.get("cycle_frequency", ""),
                "reasoning": mr.get("reasoning", ""),
            })

    print(f"  💾 Reduced suite appended to master → {master_path}")


def append_savings_summary(reduced_data: dict,
                            summary_path: str = "outputs/energy_savings_summary.csv"):
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    s = reduced_data["summary"]

    exists = os.path.exists(summary_path)
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_CSV_COLUMNS)
        if not exists:
            w.writeheader()
        w.writerow(s)

    print(f"  💾 Savings summary appended → {summary_path}")
    return summary_path