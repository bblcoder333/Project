import json
import re
import csv
import os
import torch
from json_repair import repair_json

# ─── CSV columns ──────────────────────────────────────────────────────────────

MR_CSV_COLUMNS = [
    "mr_id",
    "source_tc_id",
    "screen_id",
    "topic",
    "mr_category",
    "transformation",
    "follow_up_steps",
    "follow_up_test_data",
    "expected_relation",
    "confidence",
]

# Fields that are never meaningful for monotonicity / input transformation -
# these are identifiers/demographics, not measurements that scale meaningfully.
# Generalized as a SKIP list rather than an ALLOW list so new field names on
# unseen screens (Weight, Score, Distance, Price, Quantity, Calories, Speed,
# Duration, etc.) are recognized automatically without needing to be enumerated.
_SKIP_NUMERIC_FIELDS = {
    "age", "height", "gender", "rank", "year", "id", "xp", "phone", "number",
    "code", "pin", "otp", "zip", "zipcode", "postal", "version", "index",
    "position", "page", "count", "quantity_id", "user id", "order id",
}

# ─── Prompt ───────────────────────────────────────────────────────────────────

MR_PROMPT_TEMPLATE = """
You are Agent 3, a Metamorphic Testing Agent. Generate follow-up metamorphic
relations (MRs) from the source test cases below.

SCREEN ID: {screen_id}
TOPIC: {topic}

Classify each test case into exactly ONE pattern below (A-F) and generate the
matching MR(s) using the EXACT JSON shape shown. Patterns:

A. INVALID INPUT - type is Negative, OR title/expected mentions "invalid"/"error".
   -> ONE VALIDATION_CONSISTENCY MR: change to a DIFFERENT invalid value
   (e.g. -1 -> -5). Same validation error should still appear.
   THIS PATTERN HAS NO EXCEPTIONS: every single Negative test case, on every
   field (age, waist, height, or any other), gets its own separate
   VALIDATION_CONSISTENCY MR. Never substitute ROBUSTNESS for a Negative test
   case. Never skip one because another field already has one.
   Example: TC_124_003, test_data="Age: -1", steps="Step 1: Enter '-1' into the age field"
     -> {{"mr_id":"MR-01","source_tc_id":"TC_124_003","mr_category":"VALIDATION_CONSISTENCY",
       "transformation":"Change invalid value from -1 to -5",
       "follow_up_steps":["Step 1: Enter '-5' into the age field"],
       "follow_up_test_data":"Invalid value: -5",
       "expected_relation":"Same validation error should appear as with the original invalid value",
       "confidence":"high"}}

B. UNIT TOGGLE - test_data or steps mention "CM | KG" or "IN | LB".
   -> ONE INVARIANCE MR: switch to the other unit. Ratio/result must stay
   approximately the same - never combine with an invalid value.
   Example: TC_124_004, steps="Step 1: Tap the 'CM | KG' button"
     -> {{"mr_id":"MR-02","source_tc_id":"TC_124_004","mr_category":"INVARIANCE",
       "transformation":"Switch unit from CM | KG to IN | LB",
       "follow_up_steps":["Step 1: Tap the 'CM | KG' button","Step 2: Tap the 'IN | LB' button"],
       "follow_up_test_data":"Unit: IN | LB",
       "expected_relation":"Computed ratio remains approximately the same after unit conversion",
       "confidence":"high"}}

C. TAB / SEGMENT - steps mention tapping a named tab (e.g. "Tap the 'TOTAL' tab").
   -> ONE INVARIANCE MR: switch away to a different tab, then back. Content/state
   of the original tab must be unchanged. This TC will ALSO separately receive
   its own ROBUSTNESS MR - generating the INVARIANCE MR here does not exempt it.
   Example: TC_295_001, steps="Step 1: Tap the 'THIS COURSE' tab"
     -> {{"mr_id":"MR-03","source_tc_id":"TC_295_001","mr_category":"INVARIANCE",
       "transformation":"Switch away from 'THIS COURSE' and back",
       "follow_up_steps":["Step 1: Tap the 'THIS COURSE' tab","Step 2: Tap a different tab, then tap 'THIS COURSE' again"],
       "follow_up_test_data":"Same as source test",
       "expected_relation":"Content and state of 'THIS COURSE' remain the same after switching away and back",
       "confidence":"medium"}}

D. VALID NUMERIC MEASUREMENT - test_data has a VALID (non-negative) number for
   any MEASURABLE quantity field (e.g. Waist, Weight, Score, Distance, Price,
   Quantity, Calories, Speed, Duration, or similar). Never Age, Height, Gender,
   or any ID/phone/code/zip/pin/version number - those are identifiers or
   demographics, not measurements that scale meaningfully. Both values used
   must be VALID and POSITIVE - never pair with an invalid value, that belongs
   to pattern A.
   -> TWO MRs from the SAME valid increase: MONOTONICITY and INPUT_TRANSFORMATION.
   THIS PATTERN HAS NO EXCEPTIONS: every test case with a qualifying numeric
   field (Waist, Weight, or Score) must produce BOTH MRs. Do not skip this
   pattern even if it is the only numeric test case on the screen.
   Example: TC_124_006, test_data="Waist: 6 cm", steps="Step 1: Enter '6' into the waist field"
     -> {{"mr_id":"MR-04","source_tc_id":"TC_124_006","mr_category":"MONOTONICITY",
       "transformation":"Increase Waist from 6 to 26",
       "follow_up_steps":["Step 1: Enter '26' into the waist field"],
       "follow_up_test_data":"Waist: 26 cm",
       "expected_relation":"Computed ratio should be greater than with Waist=6",
       "confidence":"high"}}
     and -> {{"mr_id":"MR-05","source_tc_id":"TC_124_006","mr_category":"INPUT_TRANSFORMATION",
       "transformation":"Change Waist from 6 to 26",
       "follow_up_steps":["Step 1: Enter '26' into the waist field"],
       "follow_up_test_data":"Waist: 26 cm",
       "expected_relation":"Computation completes successfully with updated output values",
       "confidence":"high"}}

E. FUNCTIONAL TEST WITH A TAP/CLICK ACTION - has a Tap, Click, or Select step.
   -> ONE ROBUSTNESS MR: repeat the SAME action a second time. App should not
   crash. follow_up_steps MUST include the original step(s) AND the repeat.
   APPLY THIS TO EVERY test case with a tap/click action, INCLUDING test cases
   that also matched pattern B, C, or F - a TC can and should receive BOTH its
   pattern B/C/F MR AND its own separate pattern E robustness MR. Do NOT apply
   this to a Negative test case - those belong to pattern A only.
   Example: TC_001, steps="Step 1: Tap the 'GET STARTED' button"
     -> {{"mr_id":"MR-06","source_tc_id":"TC_001","mr_category":"ROBUSTNESS",
       "transformation":"Repeat the primary action a second time",
       "follow_up_steps":["Step 1: Tap the 'GET STARTED' button","Step 2: Tap the 'GET STARTED' button (repeat a second time)"],
       "follow_up_test_data":"Same as source test",
       "expected_relation":"App does not crash or show an inconsistent state on repeated action",
       "confidence":"high"}}

F. 3+ SEPARATE test cases that each tap ONE icon/tab/button/card of a similar
   kind (e.g. 3 different social media icons, even if they appear later in the list).
   -> ONLY ONE INTERACTION_CONSISTENCY MR total for the whole group of 3+,
   anchored to the FIRST one's source_tc_id. Do not generate a separate copy of
   this MR for each of the 3 test cases - it is one MR, not three.
   Example: TC_005 "Tap Facebook icon", TC_006 "Tap LinkedIn icon", TC_007 "Tap Google+ icon"
     -> {{"mr_id":"MR-07","source_tc_id":"TC_005","mr_category":"INTERACTION_CONSISTENCY",
       "transformation":"Reverse tap order across 3 icon interactions",
       "follow_up_steps":["Step 1: Tap the Google+ icon","Step 2: Tap the LinkedIn icon","Step 3: Tap the Facebook icon"],
       "follow_up_test_data":"Same as source tests",
       "expected_relation":"UI remains stable and no crash or inconsistent state occurs",
       "confidence":"high"}}
   IMPORTANT: ALL 3 test cases in the group - INCLUDING the anchor TC_005 -
   STILL ALSO each get their own separate ROBUSTNESS MR from pattern E. The
   anchor TC is not exempt from pattern E just because it anchors this MR.
   That means 3 source test cases here produce 4 total MRs: 3 ROBUSTNESS
   (one per icon) + 1 INTERACTION_CONSISTENCY (the group).

CRITICAL RULES:
- A test case with ONLY "Observe" steps (no Tap/Click/Enter/Select) matches NO
  pattern. Generate ZERO MRs for it.
- Skip Accessibility-type test cases entirely (zero MRs) unless they mention
  "invalid"/"error" (pattern A only).
- Every output field must contain real, specific text - never leave any field
  blank or copy just one bare value into it.
- Use exact field names, labels, and values from the source test case. Never
  invent a value not literally present in the source test case.
- follow_up_steps: strip old "Step N:" prefixes, renumber from "Step 1:".
- Patterns are not mutually exclusive with pattern E: a TC matching B, C, or F
  ALSO gets its own pattern E robustness MR, unless it is Accessibility or
  Negative type or observe-only.

---

SOURCE TEST CASES - there are exactly {tc_count} test cases below, numbered
[1] through [{tc_count}]. Of these, exactly {negative_count} are type=Negative,
and exactly {numeric_count} have a valid numeric Waist/Weight/Score field.

You must produce at least one pattern decision for EVERY numbered test case,
including the LAST one, [{tc_count}]. You must produce EXACTLY {negative_count}
VALIDATION_CONSISTENCY MRs (pattern A) and EXACTLY {numeric_count} pairs of
MONOTONICITY + INPUT_TRANSFORMATION MRs (pattern D) - no more, no fewer.
Before you finish, count both totals and confirm they match. If they do not
match, find the missing test case and add its MR(s) before finishing.

{source_csv}

---

Work through test cases [1] to [{tc_count}] IN ORDER. Do not stop early.
Remember: every tap/click TC gets its own ROBUSTNESS MR even if it also got
an INVARIANCE, MONOTONICITY/INPUT_TRANSFORMATION, or INTERACTION_CONSISTENCY
MR from another pattern.

OUTPUT ONLY valid JSON, no markdown, no explanation:

{{
  "screen_id": "{screen_id}",
  "metamorphic_relations": [ ... one object per generated MR ... ]
}}
"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _tc_data_to_csv_string(tc_data: dict) -> str:
    """
    Convert Agent 2 test cases dict to a :: separated string for the prompt.
    Each row is prefixed with [N] so the model has an explicit, checkable
    counter, referenced by the "work through [1] to [tc_count]" instruction.
    """
    lines = []
    for i, tc in enumerate(tc_data.get("test_cases", []), start=1):
        steps = " | ".join(tc.get("steps", []))
        test_data = tc.get("test_data", [])
        test_data_str = " | ".join(test_data) if isinstance(test_data, list) else str(test_data)
        expected = " | ".join(tc.get("expected", []))
        lines.append(
            f"[{i}] {tc.get('test_case_id','')} :: {tc.get('title','')} :: "
            f"{tc.get('type','')} :: {test_data_str} :: {steps} :: {expected}"
        )
    return "\n".join(lines)


def _count_negative_tcs(tc_data: dict) -> int:
    """Count test cases with type == 'Negative'."""
    return sum(
        1 for tc in tc_data.get("test_cases", [])
        if tc.get("type", "").strip().lower() == "negative"
    )


def _extract_qualifying_numeric_field(tc: dict):
    """
    Return (field_name, value) if this TC has a VALID positive numeric value
    for Waist, Weight, or Score (never Age/Height/Gender/ID/phone). Looks in
    test_data first, then falls back to scanning steps text.
    Returns None if no qualifying field is found.
    """
    test_data = tc.get("test_data", [])
    test_data_str = " | ".join(test_data) if isinstance(test_data, list) else str(test_data)
    steps_str = " ".join(tc.get("steps", []))

    # Source 1: structured test_data e.g. "Waist: 6 cm"
    for segment in re.split(r"\s*\|\s*|,\s*", test_data_str):
        m = re.search(r"([A-Za-z ]+):\s*([-\d.]+)", segment)
        if m:
            field = m.group(1).strip()
            val = m.group(2).strip()
            clean_field = re.sub(r"\s*(field|input|value|measurement)\s*$", "", field, flags=re.IGNORECASE).strip()
            if clean_field.lower() in _SKIP_NUMERIC_FIELDS:
                continue
            try:
                num = float(val)
            except ValueError:
                continue
            if num <= 0:
                continue
            if len(val.replace(".", "")) > 7:  # skip phone numbers / long IDs
                continue
            return (clean_field, val)

    # Source 2: step text e.g. "Enter the waist measurement as 6 cm"
    p1 = re.compile(
        r"enter(?:\s+the)?\s+([a-zA-Z ]+?)\s+(?:measurement\s+)?as\s+([0-9]+(?:\.[0-9]+)?)",
        re.IGNORECASE,
    )
    p2 = re.compile(
        r"enter\s+'([0-9]+(?:\.[0-9]+)?)'\s+into\s+(?:the\s+)?([a-zA-Z ]+)",
        re.IGNORECASE,
    )
    m1 = p1.search(steps_str)
    if m1:
        field = m1.group(1).strip()
        val = m1.group(2).strip()
        if field.lower() not in _SKIP_NUMERIC_FIELDS and float(val) > 0 and len(val.replace(".", "")) <= 7:
            return (field, val)
    m2 = p2.search(steps_str)
    if m2:
        val = m2.group(1).strip()
        field = re.sub(r"\s*(field|input|value)\s*$", "", m2.group(2).strip(), flags=re.IGNORECASE).strip()
        if field.lower() not in _SKIP_NUMERIC_FIELDS and float(val) > 0 and len(val.replace(".", "")) <= 7:
            return (field, val)

    return None


def _count_numeric_qualifying_tcs(tc_data: dict) -> int:
    """Count test cases that qualify for Pattern D (used in the prompt checkpoint)."""
    count = 0
    for tc in tc_data.get("test_cases", []):
        if tc.get("type", "").strip().lower() != "functional":
            continue
        if _extract_qualifying_numeric_field(tc) is not None:
            count += 1
    return count


def _safe_join(val):
    if not val:
        return ""
    if isinstance(val, str):
        return val
    return ", ".join(
        str(v) if not isinstance(v, dict) else json.dumps(v)
        for v in val
    )


def _build_tc_lookup(tc_data: dict) -> dict:
    return {tc["test_case_id"]: tc for tc in tc_data.get("test_cases", []) if "test_case_id" in tc}


def _clean_and_renumber(steps: list) -> list:
    """Strip any existing Step N: prefixes and renumber cleanly."""
    stripped = [re.sub(r"^Step\s*\d+:\s*", "", s, flags=re.IGNORECASE).strip() for s in steps]
    return [f"Step {i+1}: {s}" for i, s in enumerate(stripped) if s]


def _value_exists_in_tc(value: str, tc: dict) -> bool:
    """Check if a quoted/named value actually appears in the source TC's data
    as a whole word/phrase match, not a loose substring (so 'Jen' does not
    falsely match inside 'Jennifer')."""
    if not value:
        return True
    haystack = " ".join([
        tc.get("title", ""),
        " ".join(tc.get("test_data", [])) if isinstance(tc.get("test_data"), list) else str(tc.get("test_data", "")),
        " ".join(tc.get("steps", [])),
        " ".join(tc.get("expected", [])),
    ]).lower()
    pattern = r"(?<![a-z0-9])" + re.escape(value.lower()) + r"(?![a-z0-9])"
    return bool(re.search(pattern, haystack))


def _is_observe_only_tc(tc: dict) -> bool:
    """
    True if every step in this TC is a pure observation with no actionable
    verb (tap/click/select/enter/swipe/scroll). MRs should never be generated
    for these.
    """
    steps = tc.get("steps", [])
    if not steps:
        return False
    has_action = any(
        re.search(r"\b(tap|click|select|enter|swipe|scroll)\b", s, re.IGNORECASE)
        for s in steps
    )
    return not has_action


def _is_negative_tc(tc: dict) -> bool:
    """True if this TC's type is Negative."""
    return tc.get("type", "").strip().lower() == "negative"


def _validate_and_clean_mrs(mrs: list, tc_lookup: dict) -> list:
    """
    Post-processing safety net - enforces hard guarantees the prompt
    may not always achieve, mirroring the reliability of the rule-based version.
    """
    valid_categories = {
        "INVARIANCE", "MONOTONICITY", "INPUT_TRANSFORMATION",
        "VALIDATION_CONSISTENCY", "INTERACTION_CONSISTENCY", "ROBUSTNESS",
    }
    cleaned = []
    per_tc_count = {}

    for mr in mrs:
        source_tc_id = mr.get("source_tc_id", "")
        category = mr.get("mr_category", "").upper().strip()

        # Drop MRs referencing a TC that doesn't exist (hallucinated source)
        if source_tc_id not in tc_lookup and not source_tc_id.startswith("UI_DESC_"):
            continue

        tc = tc_lookup.get(source_tc_id)

        # Observe-only TCs never get any MR.
        if tc and _is_observe_only_tc(tc):
            continue

        # A Negative TC must never end up as ROBUSTNESS.
        if tc and _is_negative_tc(tc) and category == "ROBUSTNESS":
            continue

        # Drop MRs with invalid category names
        if category not in valid_categories:
            continue
        mr["mr_category"] = category

        tc_type = tc.get("type", "").lower() if tc else ""

        # Accessibility TCs only ever produce VALIDATION_CONSISTENCY (if applicable)
        if tc_type == "accessibility" and category != "VALIDATION_CONSISTENCY":
            continue

        # Block INVARIANCE / MONOTONICITY / INPUT_TRANSFORMATION on non-qualifying TCs
        if category in ("INVARIANCE", "MONOTONICITY", "INPUT_TRANSFORMATION") and tc:
            combined = " ".join(
                tc.get("steps", []) +
                (tc.get("test_data", []) if isinstance(tc.get("test_data"), list) else [str(tc.get("test_data", ""))])
            )
            has_unit_toggle = "|" in combined
            has_tab_tap = _has_tab_tap(tc) is not None
            if category == "INVARIANCE" and not (has_unit_toggle or has_tab_tap):
                continue
            if category in ("MONOTONICITY", "INPUT_TRANSFORMATION"):
                if _extract_qualifying_numeric_field(tc) is None:
                    continue

        # Hallucination guard for VALIDATION_CONSISTENCY / MONOTONICITY / INPUT_TRANSFORMATION
        if category in ("VALIDATION_CONSISTENCY", "MONOTONICITY", "INPUT_TRANSFORMATION") and tc:
            tc_test_data = tc.get("test_data", [])
            tc_test_data_str = " ".join(tc_test_data) if isinstance(tc_test_data, list) else str(tc_test_data)
            tc_steps_str = " ".join(tc.get("steps", []))
            has_enter_action = bool(re.search(r"\benter\b", tc_steps_str, re.IGNORECASE))
            has_real_value = bool(tc_test_data_str.strip()) or bool(re.search(r"\d", tc_steps_str))
            if not has_enter_action or not has_real_value:
                continue

            transformation = mr.get("transformation", "")
            from_match = re.search(r"from\s+'?([^' \n]+)'?\s+to", transformation, re.IGNORECASE)
            if from_match:
                original_value = from_match.group(1)
                if not _value_exists_in_tc(original_value, tc):
                    continue

        # Enforce cap of 3 MRs per source TC (Pattern D produces 2 + a robustness
        # backfill can add a 3rd; INTERACTION_CONSISTENCY is exempt from this cap)
        if category != "INTERACTION_CONSISTENCY":
            count = per_tc_count.get(source_tc_id, 0)
            if count >= 3:
                continue
            per_tc_count[source_tc_id] = count + 1

        # Clean and renumber steps
        steps = mr.get("follow_up_steps", [])
        if isinstance(steps, str):
            steps = [steps]
        mr["follow_up_steps"] = _clean_and_renumber(steps)

        # Fix empty/missing follow_up_test_data
        if not mr.get("follow_up_test_data", "").strip():
            if category == "INTERACTION_CONSISTENCY":
                mr["follow_up_test_data"] = "Same as source tests"
            else:
                mr["follow_up_test_data"] = "Same as source test"

        # Fix 6: ROBUSTNESS-specific data-fidelity check. ROBUSTNESS just repeats
        # the source TC's own action, so follow_up_test_data must literally be
        # "Same as source test" and follow_up_steps must reflect THIS TC's own
        # steps (not another TC's value the LLM mixed up). If either field
        # contains a numeric value that does NOT match the source TC's own
        # test_data, the LLM has cross-contaminated this MR with another TC's
        # data — discard it; backfill_dual_coverage_robustness_mrs will
        # regenerate a correct one from the source TC directly if needed.
        if category == "ROBUSTNESS" and tc:
            mr["follow_up_test_data"] = "Same as source test"
            # Strip "Step N:" prefixes before comparing numbers, so the step
            # counter itself (Step 1, Step 2...) is never mistaken for a
            # contaminated input value copied in from another TC.
            source_steps_stripped = [
                re.sub(r"^Step\s*\d+:\s*", "", s, flags=re.IGNORECASE) for s in tc.get("steps", [])
            ]
            mr_steps_stripped = [
                re.sub(r"^Step\s*\d+:\s*", "", s, flags=re.IGNORECASE) for s in mr.get("follow_up_steps", [])
            ]
            source_numbers = set(re.findall(r"-?\d+(?:\.\d+)?", " ".join(source_steps_stripped)))
            mr_numbers = set(re.findall(r"-?\d+(?:\.\d+)?", " ".join(mr_steps_stripped)))
            # Any number appearing in the MR's steps that never appeared in the
            # source TC's own steps means data leaked in from elsewhere.
            if mr_numbers - source_numbers:
                continue

        # Fix 7: catch follow_up_test_data that holds expected-result text
        # instead of test data (LLM field-shuffling). If the value matches the
        # source TC's expected_result almost verbatim, replace it with the
        # correct default rather than keep the misplaced text.
        if tc:
            expected_str = " ".join(tc.get("expected", [])).strip().lower()
            current_test_data = mr.get("follow_up_test_data", "").strip().lower()
            if expected_str and current_test_data == expected_str:
                mr["follow_up_test_data"] = (
                    "Same as source tests" if category == "INTERACTION_CONSISTENCY"
                    else "Same as source test"
                )

        mr.setdefault("expected_relation", "")
        mr.setdefault("confidence", "high")

        cleaned.append(mr)

    return cleaned


def _deduplicate_mrs(mrs: list) -> list:
    """
    Remove duplicate MRs. INTERACTION_CONSISTENCY is deduplicated by
    transformation text alone (regardless of source_tc_id), since that
    pattern should only ever produce ONE MR per group. All other categories
    are deduplicated by source_tc_id + category + transformation (category
    is included so a TC's MONOTONICITY and INPUT_TRANSFORMATION MRs - which
    can share very similar transformation text - are never mistaken for
    duplicates of each other).
    """
    seen = set()
    unique = []
    for mr in mrs:
        category = mr.get("mr_category", "")
        if category == "INTERACTION_CONSISTENCY":
            key = f"INTERACTION_CONSISTENCY|{mr.get('transformation','')}"
        else:
            key = f"{mr.get('source_tc_id','')}|{category}|{mr.get('transformation','')}"
        if key not in seen:
            seen.add(key)
            unique.append(mr)
    return unique


# ─── Shared builder for code-generated ROBUSTNESS MRs ────────────────────────

def _build_robustness_mr(tc: dict):
    """
    Build a standard ROBUSTNESS MR for a TC directly in code, mirroring the
    exact shape the LLM produces for pattern E. Confidence is "high" since
    this pattern is deterministic and unambiguous by definition.
    """
    tc_id = tc.get("test_case_id", "")
    steps = tc.get("steps", [])

    primary = None
    for step in reversed(steps):
        if re.search(r"\b(tap|click)\b", step, re.IGNORECASE):
            primary = step
            break
    if primary is None:
        return None

    stripped_primary = re.sub(r"^Step\s*\d+:\s*", "", primary, flags=re.IGNORECASE).strip()
    new_steps = _clean_and_renumber(
        steps + [f"{stripped_primary} (repeat a second time)"]
    )

    return {
        "mr_id": "",
        "source_tc_id": tc_id,
        "mr_category": "ROBUSTNESS",
        "transformation": "Repeat the primary action a second time",
        "follow_up_steps": new_steps,
        "follow_up_test_data": "Same as source test",
        "expected_relation": "App does not crash or show an inconsistent state on repeated action",
        "confidence": "high",
    }


# ─── Backfill: tab-invariance (Pattern C) ────────────────────────────────────

def _has_tab_tap(tc: dict):
    """
    Returns the quoted segment/tab name if this TC's steps tap a named
    tab-like UI element, else None. Generalized beyond the literal word "tab"
    to also catch segment, filter, category, view, and section selectors -
    common alternate terms Agent 2 may use to describe the same UI pattern
    on screens outside the original 3 examples.
    """
    segment_words = r"(?:tab|segment|filter|category|section|view|toggle)"
    for step in tc.get("steps", []):
        # "Tap the 'X' tab" / "Tap 'X' segment" / "Select the 'X' filter"
        m = re.search(
            rf"(?:tap|select|click) (?:the\s+)?'([^']+)'\s*{segment_words}",
            step, re.IGNORECASE,
        )
        if m:
            return m.group(1)
        # "Tap the X tab" without quotes around the name (last word before the segment word)
        m = re.search(
            rf"(?:tap|select|click) (?:the\s+)?([A-Za-z0-9 ]+?)\s+{segment_words}\b",
            step, re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
    return None


def backfill_tab_invariance_mrs(mrs: list, tc_data: dict) -> list:
    """Deterministic safety net for Pattern C (tab/segment invariance)."""
    covered_tc_ids = {
        mr.get("source_tc_id", "") for mr in mrs if mr.get("mr_category") == "INVARIANCE"
    }

    for tc in tc_data.get("test_cases", []):
        tc_id = tc.get("test_case_id", "")
        if tc_id in covered_tc_ids:
            continue
        if _is_observe_only_tc(tc):
            continue

        tab_name = _has_tab_tap(tc)
        if not tab_name:
            continue

        original_steps = tc.get("steps", [])
        new_steps = _clean_and_renumber(
            original_steps + [f"Tap a different tab, then tap '{tab_name}' again to return"]
        )

        mrs.append({
            "mr_id": "",
            "source_tc_id": tc_id,
            "mr_category": "INVARIANCE",
            "transformation": f"Switch away from '{tab_name}' and back",
            "follow_up_steps": new_steps,
            "follow_up_test_data": "Same as source test",
            "expected_relation": f"Content and state of '{tab_name}' remain the same after switching away and back",
            "confidence": "medium",
        })

    return mrs


# ─── Backfill: validation-consistency (Pattern A) ────────────────────────────

def backfill_validation_consistency_mrs(mrs: list, tc_data: dict) -> list:
    """
    Deterministic safety net for Pattern A (validation consistency). For every
    Negative TC that ended up with NO VALIDATION_CONSISTENCY MR at all,
    generate the correct MR directly in code. Handles both numeric invalid
    values (-1 -> -5) and text invalid values ('abc' -> '###').
    """
    covered_negative_tc_ids = {
        mr.get("source_tc_id", "")
        for mr in mrs
        if mr.get("mr_category", "") == "VALIDATION_CONSISTENCY"
    }

    for tc in tc_data.get("test_cases", []):
        tc_id = tc.get("test_case_id", "")
        if not _is_negative_tc(tc):
            continue
        if tc_id in covered_negative_tc_ids:
            continue

        test_data = tc.get("test_data", [])
        test_data_str = " ".join(test_data) if isinstance(test_data, list) else str(test_data)
        steps = tc.get("steps", [])
        steps_str = " ".join(steps)

        # Case 1: numeric invalid value
        neg_match = re.search(r"(-\d+(?:\.\d+)?)", test_data_str) or re.search(r"(-\d+(?:\.\d+)?)", steps_str)
        if neg_match:
            neg_val = neg_match.group(1)
            new_val = str(int(float(neg_val)) - 4)
            new_steps = [re.sub(re.escape(neg_val), new_val, s, count=1) for s in steps]
            new_steps = _clean_and_renumber(new_steps) if new_steps else [
                f"Repeat source test steps with value changed to {new_val}"
            ]
            mrs.append({
                "mr_id": "",
                "source_tc_id": tc_id,
                "mr_category": "VALIDATION_CONSISTENCY",
                "transformation": f"Change invalid value from {neg_val} to {new_val}",
                "follow_up_steps": new_steps,
                "follow_up_test_data": f"Invalid value: {new_val}",
                "expected_relation": "Same validation error should appear as with the original invalid value",
                "confidence": "medium",
            })
            continue

        # Case 2: text invalid value, e.g. test_data="abc"
        text_match = re.search(r"'([^']{1,20})'", test_data_str) or re.search(
            r"(?<![\w'])([a-zA-Z]{2,20})(?![\w'])", test_data_str
        )
        if text_match:
            original = text_match.group(1)
            if not re.fullmatch(r"[\d.]+", original):
                new_steps = []
                for step in steps:
                    replaced = re.sub(rf"'{re.escape(original)}'", "'###'", step, count=1)
                    if replaced == step:
                        replaced = re.sub(rf"\b{re.escape(original)}\b", "###", step, count=1)
                    new_steps.append(replaced)
                new_steps = _clean_and_renumber(new_steps) if new_steps else [
                    "Repeat source test steps with invalid value changed to '###'"
                ]
                mrs.append({
                    "mr_id": "",
                    "source_tc_id": tc_id,
                    "mr_category": "VALIDATION_CONSISTENCY",
                    "transformation": f"Change invalid text from '{original}' to '###'",
                    "follow_up_steps": new_steps,
                    "follow_up_test_data": "Invalid value: '###'",
                    "expected_relation": "Same validation error should appear as with the original invalid input",
                    "confidence": "medium",
                })

    return mrs


# ─── Backfill: numeric monotonicity / input transformation (Pattern D) ───────

def backfill_numeric_transformation_mrs(mrs: list, tc_data: dict) -> list:
    """
    Deterministic safety net for Pattern D (MONOTONICITY + INPUT_TRANSFORMATION).
    This is the category most often dropped entirely by the LLM. For every
    Functional TC with a qualifying numeric field (Waist, Weight, or Score -
    never Age/Height/Gender), ensure BOTH MRs exist.
    """
    covered_tc_ids = {
        mr.get("source_tc_id", "")
        for mr in mrs
        if mr.get("mr_category", "") in ("MONOTONICITY", "INPUT_TRANSFORMATION")
    }

    for tc in tc_data.get("test_cases", []):
        tc_id = tc.get("test_case_id", "")
        if tc.get("type", "").strip().lower() != "functional":
            continue
        if tc_id in covered_tc_ids:
            continue

        field_info = _extract_qualifying_numeric_field(tc)
        if field_info is None:
            continue
        field, val = field_info

        try:
            old = float(val)
        except ValueError:
            continue

        new = old + 20
        old_str = str(int(old)) if old == int(old) else str(old)
        new_str = str(int(new)) if new == int(new) else str(new)

        steps = tc.get("steps", [])
        new_steps = []
        for step in steps:
            replaced = re.sub(rf"\b{re.escape(old_str)}\b", new_str, step, count=1)
            new_steps.append(replaced)
        new_steps = _clean_and_renumber(new_steps) if new_steps else [
            f"Enter '{new_str}' into the {field.lower()} field"
        ]

        mrs.append({
            "mr_id": "",
            "source_tc_id": tc_id,
            "mr_category": "MONOTONICITY",
            "transformation": f"Increase {field} from {old_str} to {new_str}",
            "follow_up_steps": new_steps,
            "follow_up_test_data": f"{field}: {new_str}",
            "expected_relation": f"Computed ratio should be greater than with {field}={old_str}",
            "confidence": "high",
        })
        mrs.append({
            "mr_id": "",
            "source_tc_id": tc_id,
            "mr_category": "INPUT_TRANSFORMATION",
            "transformation": f"Change {field} from {old_str} to {new_str}",
            "follow_up_steps": list(new_steps),
            "follow_up_test_data": f"{field}: {new_str}",
            "expected_relation": "Computation completes successfully with updated output values",
            "confidence": "high",
        })

    return mrs


# ─── Backfill: dual-coverage robustness gaps ─────────────────────────────────

def backfill_dual_coverage_robustness_mrs(mrs: list, tc_lookup: dict) -> list:
    """
    Ensures every TC that received a non-ROBUSTNESS MR (INVARIANCE,
    MONOTONICITY, INPUT_TRANSFORMATION, or is the anchor of an
    INTERACTION_CONSISTENCY group) ALSO has its own independent ROBUSTNESS MR,
    matching the rule-based version's behavior of allowing multiple MR
    categories per source TC. This single function replaces what were
    previously three separate near-duplicate backfill functions.
    """
    tc_ids_needing_robustness = set()
    for mr in mrs:
        category = mr.get("mr_category", "")
        if category in ("INVARIANCE", "MONOTONICITY", "INPUT_TRANSFORMATION", "INTERACTION_CONSISTENCY"):
            tc_ids_needing_robustness.add(mr.get("source_tc_id", ""))

    tc_ids_with_robustness = {
        mr.get("source_tc_id", "") for mr in mrs if mr.get("mr_category", "") == "ROBUSTNESS"
    }

    for tc_id in tc_ids_needing_robustness:
        if tc_id in tc_ids_with_robustness:
            continue
        tc = tc_lookup.get(tc_id)
        if not tc or _is_observe_only_tc(tc) or _is_negative_tc(tc):
            continue
        new_mr = _build_robustness_mr(tc)
        if new_mr is not None:
            mrs.append(new_mr)
            tc_ids_with_robustness.add(tc_id)  # avoid double-adding within this loop

    return mrs



# ─── Backfill: interaction consistency (Pattern F) ───────────────────────────

def backfill_interaction_consistency_mrs(mrs: list, tc_data: dict) -> list:
    """
    Conservative deterministic safety net for Pattern F (INTERACTION_CONSISTENCY).
    Generalizes the original rule-based detection: groups single-tap Functional
    TCs by a shared theme keyword (icon, tab, button, card, item, option) and,
    if 3 or more share the same keyword and none of them already has an
    INTERACTION_CONSISTENCY MR, generates ONE MR reversing their tap order.

    This only runs if the LLM produced ZERO INTERACTION_CONSISTENCY MRs for
    the whole screen, since Pattern F grouping is inherently more subjective
    than the other patterns and a partially-correct LLM grouping is generally
    more trustworthy than a fully mechanical one - this backfill exists purely
    to avoid a screen with an obvious icon/button group getting zero coverage
    on an unfamiliar screen layout.
    """
    if any(mr.get("mr_category") == "INTERACTION_CONSISTENCY" for mr in mrs):
        return mrs  # LLM already handled this pattern - do not second-guess it

    theme_words = ("icon", "tab", "button", "card", "item", "option", "chip")
    single_tap_tcs = []
    for tc in tc_data.get("test_cases", []):
        if tc.get("type", "").strip().lower() != "functional":
            continue
        steps = tc.get("steps", [])
        taps = [s for s in steps if re.search(r"\b(tap|click|select)\b", s, re.IGNORECASE)]
        if len(taps) == 1:
            single_tap_tcs.append((tc.get("test_case_id", ""), taps[0]))

    if len(single_tap_tcs) < 3:
        return mrs

    theme_groups: dict = {}
    for tc_id, step in single_tap_tcs:
        for word in theme_words:
            if re.search(rf"\b{word}\b", step, re.IGNORECASE):
                theme_groups.setdefault(word, []).append((tc_id, step))
                break  # only count each TC under its first matching theme word

    for theme, group in theme_groups.items():
        if len(group) < 3:
            continue
        subset = group[:3]
        source_ids = [t[0] for t in subset]
        orig_steps = [t[1] for t in subset]
        rev_steps = _clean_and_renumber(list(reversed(orig_steps)))

        mrs.append({
            "mr_id": "",
            "source_tc_id": source_ids[0],
            "mr_category": "INTERACTION_CONSISTENCY",
            "transformation": f"Reverse tap order across {len(subset)} {theme} interactions",
            "follow_up_steps": rev_steps,
            "follow_up_test_data": "Same as source tests",
            "expected_relation": "UI remains stable and no crash or inconsistent state occurs",
            "confidence": "medium",
        })
        break  # one group is enough for a single backfilled MR per screen

    return mrs


def _renumber_mr_ids(mrs: list) -> list:
    for i, mr in enumerate(mrs, start=1):
        mr["mr_id"] = f"MR-{i:02d}"
    return mrs


# ─── Main generation function ─────────────────────────────────────────────────

def generate_metamorphic_relations(tc_data: dict, model, tokenizer) -> dict:
    """
    Agent 3 (LLM-based): takes Agent 2 output (tc_data) and generates
    metamorphic relations using Qwen 2.5-7B, with a post-processing safety net
    that enforces the same guarantees the rule-based version had, plus
    deterministic backfills covering every one of the 6 MR categories:
      - Pattern A  (VALIDATION_CONSISTENCY)               - backfill_validation_consistency_mrs
      - Pattern B  (INVARIANCE, unit toggle)               - caught by validation filter only
      - Pattern C  (INVARIANCE, tab/segment)               - backfill_tab_invariance_mrs
      - Pattern D  (MONOTONICITY + INPUT_TRANSFORMATION)   - backfill_numeric_transformation_mrs
      - Pattern E  (ROBUSTNESS, incl. dual-coverage gaps)  - backfill_dual_coverage_robustness_mrs
      - Pattern F  (INTERACTION_CONSISTENCY)               - not backfilled (multi-TC grouping
        is too context-dependent to safely reconstruct in code; relies on the
        explicit prompt instruction and worked example)
    Every category except F now has a deterministic guarantee, so the LLM's
    output quality variance only affects F and the exact wording/confidence
    of categories elsewhere - never their presence or absence.
    """
    screen_id = tc_data.get("screen_id", "unknown")
    topic     = tc_data.get("topic", "unknown")

    source_csv     = _tc_data_to_csv_string(tc_data)
    tc_lookup      = _build_tc_lookup(tc_data)
    tc_count       = len(tc_data.get("test_cases", []))
    negative_count = _count_negative_tcs(tc_data)
    numeric_count  = _count_numeric_qualifying_tcs(tc_data)

    prompt = MR_PROMPT_TEMPLATE.format(
        screen_id=screen_id,
        topic=topic,
        source_csv=source_csv,
        tc_count=tc_count,
        negative_count=negative_count,
        numeric_count=numeric_count,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a metamorphic testing expert. "
                "Output ONLY valid JSON with no markdown or explanation."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=6144,
    ).to("cuda")

    print(f"   🔄 Generating metamorphic relations (Agent 3, LLM-based, "
          f"{tc_count} TCs, {negative_count} negative, {numeric_count} numeric)...")

    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=6000,
            do_sample=False,
            top_p=1.0,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(
        ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )

    llm_mr_count = 0
    try:
        blocks = re.findall(r"\{.*\}", raw, re.DOTALL)
        if not blocks:
            print("   ⚠️  Agent 3: No JSON block found in output - relying entirely on backfills")
            mrs = []
        else:
            candidate = max(blocks, key=len)
            repaired  = repair_json(candidate)
            result    = json.loads(repaired)
            mrs = result.get("metamorphic_relations", [])

        mrs = _validate_and_clean_mrs(mrs, tc_lookup)
        llm_mr_count = len(mrs)  # count AFTER cleaning, BEFORE backfills, for diagnostics

        mrs = _deduplicate_mrs(mrs)
        mrs = backfill_tab_invariance_mrs(mrs, tc_data)
        mrs = backfill_validation_consistency_mrs(mrs, tc_data)
        mrs = backfill_numeric_transformation_mrs(mrs, tc_data)
        mrs = backfill_dual_coverage_robustness_mrs(mrs, tc_lookup)
        mrs = backfill_interaction_consistency_mrs(mrs, tc_data)
        mrs = _deduplicate_mrs(mrs)  # second pass in case a backfill introduced a dup
        mrs = _renumber_mr_ids(mrs)

        backfilled_count = len(mrs) - llm_mr_count
        print(f"   ✅ Agent 3 generated {len(mrs)} metamorphic relations "
              f"({llm_mr_count} from LLM, {backfilled_count} backfilled)")

        return {
            "screen_id":             screen_id,
            "topic":                 topic,
            "metamorphic_relations": mrs,
        }

    except Exception as e:
        print(f"   ❌ Agent 3 JSON parse failed: {e} - relying entirely on backfills")
        mrs = []
        mrs = backfill_tab_invariance_mrs(mrs, tc_data)
        mrs = backfill_validation_consistency_mrs(mrs, tc_data)
        mrs = backfill_numeric_transformation_mrs(mrs, tc_data)
        mrs = backfill_dual_coverage_robustness_mrs(mrs, tc_lookup)
        mrs = backfill_interaction_consistency_mrs(mrs, tc_data)
        mrs = _renumber_mr_ids(mrs)
        return {
            "screen_id":             screen_id,
            "topic":                 topic,
            "metamorphic_relations": mrs,
        }


# ─── Save functions ───────────────────────────────────────────────────────────

def save_mr_data(mr_data: dict, out_dir: str = "outputs/metamorphic_relations"):
    """Save MRs to individual CSV per screen (e.g. 124_MR.csv)."""
    os.makedirs(out_dir, exist_ok=True)
    screen_id = mr_data.get("screen_id", "unknown")
    topic     = mr_data.get("topic", "unknown")
    mrs       = mr_data.get("metamorphic_relations", [])

    out_path = os.path.join(out_dir, f"{screen_id}_MR.csv")
    if os.path.exists(out_path):
        os.remove(out_path)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MR_CSV_COLUMNS)
        w.writeheader()
        for mr in mrs:
            w.writerow({
                "mr_id":               mr.get("mr_id", ""),
                "source_tc_id":        mr.get("source_tc_id", ""),
                "screen_id":           screen_id,
                "topic":               topic,
                "mr_category":         mr.get("mr_category", ""),
                "transformation":      mr.get("transformation", ""),
                "follow_up_steps":     _safe_join(mr.get("follow_up_steps", [])),
                "follow_up_test_data": mr.get("follow_up_test_data", ""),
                "expected_relation":   mr.get("expected_relation", ""),
                "confidence":          mr.get("confidence", ""),
            })

    print(f"  💾 MR data saved → {out_path}")
    return out_path


def append_mr_to_master_csv(
    mr_data: dict,
    master_path: str = "outputs/metamorphic_relations_master.csv",
):
    """Append MRs to the master MR CSV."""
    os.makedirs(os.path.dirname(master_path), exist_ok=True)
    screen_id = mr_data.get("screen_id", "unknown")
    topic     = mr_data.get("topic", "unknown")
    mrs       = mr_data.get("metamorphic_relations", [])

    exists = os.path.exists(master_path)
    with open(master_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MR_CSV_COLUMNS)
        if not exists:
            w.writeheader()
        for mr in mrs:
            w.writerow({
                "mr_id":               mr.get("mr_id", ""),
                "source_tc_id":        mr.get("source_tc_id", ""),
                "screen_id":           screen_id,
                "topic":               topic,
                "mr_category":         mr.get("mr_category", ""),
                "transformation":      mr.get("transformation", ""),
                "follow_up_steps":     _safe_join(mr.get("follow_up_steps", [])),
                "follow_up_test_data": mr.get("follow_up_test_data", ""),
                "expected_relation":   mr.get("expected_relation", ""),
                "confidence":          mr.get("confidence", ""),
            })

    print(f"  💾 MRs appended to master → {master_path}")