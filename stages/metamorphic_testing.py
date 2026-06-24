import re
import csv
import os
import json

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

# ─── Rule helpers ─────────────────────────────────────────────────────────────

def _extract_numeric_inputs(test_data: str, steps: list = None) -> list[tuple[str, str]]:
    """
    Pull out (field_name, value) pairs from test_data string AND step text.
    E.g. "Age: 26 | Waist: 100 cm" or "Enter the waist measurement as 100 cm"
    """
    pairs = []
    seen_fields = set()

    # Source 1: structured test_data e.g. "Age: 26 | Waist: 100"
    for segment in re.split(r"\s*\|\s*|,\s*", test_data):
        m = re.search(r"([A-Za-z ]+):\s*([-\d.]+)", segment)
        if m:
            field = m.group(1).strip().lower()
            val   = m.group(2).strip()
            if field not in seen_fields:
                seen_fields.add(field)
                pairs.append((m.group(1).strip(), val))

    # Source 2: step text e.g. "Enter the waist measurement as 100 cm"
    if steps:
        p1 = re.compile(
            r"enter(?:\s+the)?\s+([a-zA-Z ]+?)\s+(?:measurement\s+)?as\s+([0-9]+(?:\.[0-9]+)?)",
            re.IGNORECASE,
        )
        p2 = re.compile(
            r"enter\s+'([0-9]+(?:\.[0-9]+)?)'\s+into\s+(?:the\s+)?([a-zA-Z ]+)",
            re.IGNORECASE,
        )
        for step in steps:
            m1 = p1.search(step)
            if m1:
                field = m1.group(1).strip().lower()
                val   = m1.group(2).strip()
                if field not in seen_fields:
                    seen_fields.add(field)
                    pairs.append((m1.group(1).strip(), val))
                continue
            m2 = p2.search(step)
            if m2:
                field = m2.group(2).strip().lower()
                val   = m2.group(1).strip()
                if field not in seen_fields:
                    seen_fields.add(field)
                    pairs.append((m2.group(2).strip(), val))

    return pairs


def _has_unit_toggle(steps: list[str], test_data: str) -> bool:
    combined = " ".join(steps) + " " + test_data
    return bool(re.search(r"\bCM\s*[|/]\s*KG\b|\bIN\s*[|/]\s*LB\b|\bunit\b|\btoggle\b",
                          combined, re.IGNORECASE))


def _has_explicit_unit_pattern(steps: list[str], test_data: str) -> bool:
    """True only for the strong, explicit CM|KG / IN|LB pattern (not the generic 'unit'/'toggle' words)."""
    combined = " ".join(steps) + " " + test_data
    return bool(re.search(r"\bCM\s*[|/]\s*KG\b|\bIN\s*[|/]\s*LB\b", combined, re.IGNORECASE))


def _has_multi_click_sequence(steps: list[str], min_taps: int = 2) -> bool:
    """True if there are min_taps+ tap/click steps referencing distinct items.

    Lowered default from 3 -> 2 since most generated TCs only have 1-2 tap
    steps per test case; requiring 3 meant this almost never fired in-TC and
    we were relying entirely on the cross-TC fallback.
    """
    taps = [s for s in steps if re.search(r"\b(tap|click|select)\b", s, re.IGNORECASE)]
    return len(taps) >= min_taps


def _extract_negative_value(test_data: str) -> str | None:
    """Return the negative numeric value string if present, else None."""
    m = re.search(r"(-\d+(?:\.\d+)?)", test_data)
    return m.group(1) if m else None


def _has_tab_or_toggle_control(steps: list[str], test_data: str) -> bool:
    """True if the TC interacts with a tab, segment, or filter-style control
    (e.g. 'THIS COURSE' / 'TOTAL' tabs) that isn't a unit toggle."""
    combined = " ".join(steps) + " " + test_data
    return bool(re.search(r"\btab\b|\bsegment\b|\bfilter\b", combined, re.IGNORECASE))


def _confidence_for(tc: dict, field_name: str | None = None, basis: str = "tc") -> str:
    """
    Heuristic confidence scoring, replacing the previous hardcoded 'high' constant.

    - "high"   : field/value was explicitly present in structured test_data
    - "medium" : value had to be scraped from step text, or the signal
                 triggering the rule was a generic keyword match rather than
                 an explicit, unambiguous pattern
    - "low"    : derived from Agent 1's free-text UI description rather than
                 a concrete Agent 2 test case (handled separately in the
                 UI-description fallback path, kept as-is)
    """
    if basis == "generic_keyword":
        # e.g. invariance triggered by the word "unit"/"toggle" rather than
        # an explicit CM|KG / IN|LB pattern
        return "medium"

    test_data = " ".join(tc.get("test_data", []))
    if not test_data.strip():
        return "medium"

    if field_name and field_name.lower() not in test_data.lower():
        return "medium"

    return "high"


def _make_mr(mr_id, source_tc_id, category, transformation,
             follow_up_steps, follow_up_test_data, expected_relation, confidence):
    return {
        "mr_id":               mr_id,
        "source_tc_id":        source_tc_id,
        "mr_category":         category,
        "transformation":      transformation,
        "follow_up_steps":     follow_up_steps,
        "follow_up_test_data": follow_up_test_data,
        "expected_relation":   expected_relation,
        "confidence":          confidence,
    }


# ─── The rule applicators ─────────────────────────────────────────────────────

def _is_invalid_input_tc(tc: dict) -> bool:
    """Return True if this TC is testing invalid input, regardless of type label."""
    if tc.get("type", "").lower() in ("negative", "validation"):
        return True
    combined = " ".join([
        tc.get("title", ""),
        " ".join(tc.get("expected", [])),
    ]).lower()
    return bool(re.search(r"invalid|error message|not accept|reject", combined))


def _rule_validation_consistency(tc: dict) -> dict | None:
    """Invalid input test -> change invalid value to a different invalid value."""
    if not _is_invalid_input_tc(tc):
        return None

    test_data_str = " ".join(tc.get("test_data", []))
    steps = tc.get("steps", [])

    # Case 1: negative numeric value e.g. -1 -> -5
    neg_val = _extract_negative_value(test_data_str)
    if neg_val is not None:
        new_val = str(int(float(neg_val)) - 4)
        new_steps = []
        for step in steps:
            replaced = re.sub(r"-\d+(?:\.\d+)?", new_val, step, count=1)
            new_steps.append(replaced)
        if not new_steps:
            new_steps = [f"Repeat source test steps with value changed to {new_val}"]
        return _make_mr(
            mr_id="",
            source_tc_id=tc["test_case_id"],
            category="VALIDATION_CONSISTENCY",
            transformation=f"Change invalid value from {neg_val} to {new_val}",
            follow_up_steps=new_steps,
            follow_up_test_data=f"Invalid value: {new_val}",
            expected_relation="Same validation error should appear as with the original invalid value",
            confidence=_confidence_for(tc, basis="tc"),
        )

    # Case 2: non-numeric invalid text e.g. 'abc' or abc -> different invalid text '###'
    text_invalid = re.search(r"'([^']{1,20})'|(?<!\w)([a-zA-Z!@#$%^&*]{2,20})(?!\w)", test_data_str)
    skip_words = {"enter", "tap", "select", "click", "scroll", "verify", "open",
                  "step", "the", "and", "into", "field", "bar", "button"}
    if text_invalid:
        original = text_invalid.group(1) or text_invalid.group(2)
        if original and original.lower() not in skip_words and not re.fullmatch(r"[\d.]+", original):
            new_invalid = "###"
            new_steps = []
            for step in steps:
                replaced = re.sub(rf"'{re.escape(original)}'", f"'{new_invalid}'", step, count=1)
                if replaced == step:
                    replaced = re.sub(rf"\b{re.escape(original)}\b", new_invalid, step, count=1)
                new_steps.append(replaced)
            if not new_steps:
                new_steps = [f"Repeat source test steps with invalid value changed to '{new_invalid}'"]
            return _make_mr(
                mr_id="",
                source_tc_id=tc["test_case_id"],
                category="VALIDATION_CONSISTENCY",
                transformation=f"Change invalid text input from '{original}' to '{new_invalid}'",
                follow_up_steps=new_steps,
                follow_up_test_data=f"Invalid value: '{new_invalid}'",
                expected_relation="Same validation error should appear as with the original invalid input",
                confidence=_confidence_for(tc, basis="tc"),
            )

    return None


def _rule_input_transformation(tc: dict) -> dict | None:
    """Functional test with a numeric input -> substitute a different valid value."""
    if tc.get("type", "").lower() not in ("functional", ""):
        return None
    test_data_str = " | ".join(tc.get("test_data", []))
    pairs = _extract_numeric_inputs(test_data_str, tc.get("steps", []))
    for name, val in pairs:
        clean_name = re.sub(r'\s*(field|input|value|measurement)\s*$', '', name, flags=re.IGNORECASE).strip()
        if any(skip in clean_name.lower() for skip in ("age", "gender", "height")):
            continue
        name = clean_name
        try:
            old = float(val)
        except ValueError:
            continue
        if old < 0:
            continue
        if len(val.replace('.', '')) > 7:
            continue
        new = old + 20 if old >= 0 else old - 20
        new_steps = []
        for step in tc.get("steps", []):
            replaced = re.sub(
                rf"\b{re.escape(val)}\b", str(int(new)), step, count=1
            )
            new_steps.append(replaced)
        if not new_steps:
            new_steps = [f"Repeat source steps with {name} changed to {int(new)}"]
        return _make_mr(
            mr_id="",
            source_tc_id=tc["test_case_id"],
            category="INPUT_TRANSFORMATION",
            transformation=f"Change {name} from {int(old)} to {int(new)}",
            follow_up_steps=new_steps,
            follow_up_test_data=f"{name}: {int(new)}",
            expected_relation="Computation completes successfully with updated output values",
            confidence=_confidence_for(tc, field_name=name, basis="tc"),
        )
    return None


def _rule_monotonicity(tc: dict) -> dict | None:
    """Functional test with a positive numeric input -> increase it, ratio should increase."""
    if tc.get("type", "").lower() not in ("functional", ""):
        return None
    test_data_str = " | ".join(tc.get("test_data", []))
    pairs = _extract_numeric_inputs(test_data_str, tc.get("steps", []))
    for name, val in pairs:
        clean_name = re.sub(r'\s*(field|input|value|measurement)\s*$', '', name, flags=re.IGNORECASE).strip()
        if any(skip in clean_name.lower() for skip in ("age", "gender", "height")):
            continue
        name = clean_name
        try:
            old = float(val)
        except ValueError:
            continue
        if old <= 0:
            continue
        if len(val.replace('.', '')) > 7:
            continue
        new = old + 20
        new_steps = []
        for step in tc.get("steps", []):
            replaced = re.sub(
                rf"\b{re.escape(val)}\b", str(int(new)), step, count=1
            )
            new_steps.append(replaced)
        if not new_steps:
            new_steps = [f"Repeat source steps with {name} increased to {int(new)}"]
        return _make_mr(
            mr_id="",
            source_tc_id=tc["test_case_id"],
            category="MONOTONICITY",
            transformation=f"Increase {name} from {int(old)} to {int(new)}",
            follow_up_steps=new_steps,
            follow_up_test_data=f"{name}: {int(new)}",
            expected_relation=f"Computed ratio should be greater than with {name}={int(old)}",
            confidence=_confidence_for(tc, field_name=name, basis="tc"),
        )
    return None


def _rule_invariance(tc: dict) -> dict | None:
    """Functional test with a unit toggle -> switch unit, computed ratio should stay the same."""
    if tc.get("type", "").lower() not in ("functional", ""):
        return None
    steps = tc.get("steps", [])
    test_data_str = " | ".join(tc.get("test_data", []))
    if not _has_unit_toggle(steps, test_data_str):
        return None

    explicit = _has_explicit_unit_pattern(steps, test_data_str)

    if re.search(r"\bCM\b|\bKG\b", test_data_str, re.IGNORECASE):
        from_unit, to_unit = "CM | KG", "IN | LB"
    else:
        from_unit, to_unit = "IN | LB", "CM | KG"
    new_steps = _clean_and_renumber(steps, extra=f"Tap the '{to_unit}' toggle to switch units")
    return _make_mr(
        mr_id="",
        source_tc_id=tc["test_case_id"],
        category="INVARIANCE",
        transformation=f"Switch measurement unit from {from_unit} to {to_unit}",
        follow_up_steps=new_steps,
        follow_up_test_data=f"Unit: {to_unit}",
        expected_relation="Computed ratio remains approximately the same after unit conversion",
        confidence=_confidence_for(tc, basis="tc" if explicit else "generic_keyword"),
    )


def _rule_tab_roundtrip_invariance(tc: dict) -> dict | None:
    """
    NEW RULE: Functional test involving a tab/segment/filter control (e.g. a
    'THIS COURSE' / 'TOTAL' tab pair) -> switch away and back, the original
    view's state/content should be unchanged. This targets non-numeric
    screens (leaderboards, filters, segmented views) where the unit-toggle
    rule above never fires, so they previously fell through to ROBUSTNESS
    as the only applicable category.
    """
    if tc.get("type", "").lower() not in ("functional", ""):
        return None
    steps = tc.get("steps", [])
    test_data_str = " | ".join(tc.get("test_data", []))

    if not _has_tab_or_toggle_control(steps, test_data_str):
        return None
    # Don't double-fire on the same control the unit-invariance rule already covers
    if _has_unit_toggle(steps, test_data_str):
        return None

    # Find the tab/segment name being tapped, if quoted in the steps
    m = re.search(r"tap (?:the\s+)?'([^']+)'\s*(?:tab|segment|filter)", " ".join(steps), re.IGNORECASE)
    tab_name = m.group(1) if m else "the current tab"

    new_steps = _clean_and_renumber(
        steps,
        extra=f"Tap to switch to a different tab, then tap '{tab_name}' again to return"
    )
    return _make_mr(
        mr_id="",
        source_tc_id=tc["test_case_id"],
        category="INVARIANCE",
        transformation=f"Switch away from '{tab_name}' and back",
        follow_up_steps=new_steps,
        follow_up_test_data="Same as source test",
        expected_relation=f"Content and state of '{tab_name}' remain the same after switching away and back",
        confidence=_confidence_for(tc, basis="generic_keyword"),
    )


def _rule_interaction_consistency(tc: dict) -> dict | None:
    """Test with 2+ tap steps -> reverse the click order."""
    steps = tc.get("steps", [])
    if not _has_multi_click_sequence(steps, min_taps=2):
        return None
    tap_indices = [i for i, s in enumerate(steps)
                   if re.search(r"\b(tap|click|select)\b", s, re.IGNORECASE)]
    if len(tap_indices) < 2:
        return None
    reversed_steps = list(steps)
    tap_steps_vals = [steps[i] for i in tap_indices]
    for idx, orig_idx in enumerate(tap_indices):
        reversed_steps[orig_idx] = tap_steps_vals[-(idx + 1)]
    return _make_mr(
        mr_id="",
        source_tc_id=tc["test_case_id"],
        category="INTERACTION_CONSISTENCY",
        transformation="Reverse the order of tap/click interactions",
        follow_up_steps=reversed_steps,
        follow_up_test_data="Same as source test",
        expected_relation="UI remains stable and no crash or inconsistent state occurs",
        confidence=_confidence_for(tc, basis="tc"),
    )


def _clean_and_renumber(steps: list, extra: str = None, repeat: bool = False) -> list:
    """Strip existing Step N: prefixes and renumber cleanly, optionally appending an extra step."""
    stripped = [re.sub(r"^Step\s*\d+:\s*", "", s, flags=re.IGNORECASE).strip() for s in steps]
    if extra:
        extra_clean = re.sub(r"^Step\s*\d+:\s*", "", extra, flags=re.IGNORECASE).strip()
        if repeat:
            extra_clean = extra_clean + " (repeat a second time)"
        stripped.append(extra_clean)
    return [f"Step {i+1}: {s}" for i, s in enumerate(stripped)]


def _rule_robustness(tc: dict) -> dict | None:
    """Any functional test -> repeat the primary action twice."""
    if tc.get("type", "").lower() not in ("functional", ""):
        return None
    steps = tc.get("steps", [])
    primary = None
    for step in reversed(steps):
        if re.search(r"\b(tap|click)\b", step, re.IGNORECASE):
            primary = step
            break
    if primary is None:
        return None
    new_steps = _clean_and_renumber(steps, extra=primary, repeat=True)
    return _make_mr(
        mr_id="",
        source_tc_id=tc["test_case_id"],
        category="ROBUSTNESS",
        transformation="Repeat the primary action a second time",
        follow_up_steps=new_steps,
        follow_up_test_data="Same as source test",
        expected_relation="App does not crash or show an inconsistent state on repeated action",
        confidence=_confidence_for(tc, basis="tc"),
    )


# ─── Rule dispatch table ──────────────────────────────────────────────────────
# Each rule is tried per TC; at most one MR per rule per TC is emitted.
# Cap after a few successful rules to keep 1-3 MRs per TC.
#
# _rule_tab_roundtrip_invariance is new: it's inserted before robustness so
# that tab/segment-style screens get an INVARIANCE MR instead of falling
# through to ROBUSTNESS as their only option.

_RULES = [
    _rule_validation_consistency,       # Negative TCs first
    _rule_input_transformation,         # Functional TCs — controlled value change
    _rule_monotonicity,                 # Functional TCs — directional increase
    _rule_invariance,                   # Explicit unit-toggle TCs
    _rule_tab_roundtrip_invariance,     # NEW: tab/segment/filter round-trip TCs
    _rule_interaction_consistency,      # Multi-tap TCs (now 2+ taps, was 3+)
    _rule_robustness,                   # Any functional TC — repeat action
]


# ─── UI Description-based MR generation (Agent 1 fallback) ───────────────────

# Fields worth testing with MONOTONICITY / INPUT_TRANSFORMATION.
# Added "xp" (and a couple of common point/count synonyms) so leaderboard-
# style screens with XP values aren't invisible to this fallback.
_NUMERIC_FIELDS_OF_INTEREST = {
    "waist", "weight", "bmi", "score", "points", "distance", "calories",
    "xp", "rating", "count",
}
# Fields to skip (not meaningful to increase)
_SKIP_FIELDS = {"age", "height", "gender", "rank", "year", "id"}


def _parse_numeric_fields_from_description(description: str) -> list[tuple[str, str]]:
    """
    Extract numeric input field values from Agent 1's UI description text.
    Returns list of (field_name, value) pairs, skipping non-meaningful fields.
    """
    found = {}

    field_alt = "|".join(_NUMERIC_FIELDS_OF_INTEREST)

    patterns = [
        re.compile(
            rf'\b({field_alt})\b'
            r'\s*\(([0-9]+(?:\.[0-9]+)?)',
            re.IGNORECASE,
        ),
        re.compile(
            rf'\b({field_alt})\b'
            r'\s*\(set to\s*([0-9]+(?:\.[0-9]+)?)',
            re.IGNORECASE,
        ),
        re.compile(
            rf'\b({field_alt})\b'
            r'[^.\n]{0,30}?(?:set to|value|:)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:cm|kg|lb|feet|ft|m|xp)?',
            re.IGNORECASE,
        ),
        re.compile(
            rf'\b({field_alt})\b'
            r'\s*(?:input\s+)?field[^.\n]{0,30}?([0-9]+(?:\.[0-9]+)?)',
            re.IGNORECASE,
        ),
        re.compile(
            rf'\b({field_alt})\b'
            r'\s+measurement\s+as\s+([0-9]+(?:\.[0-9]+)?)',
            re.IGNORECASE,
        ),
        # "56876 XP" style: number BEFORE the field name (common for XP/points)
        re.compile(
            rf'([0-9]+(?:\.[0-9]+)?)\s*({field_alt})\b',
            re.IGNORECASE,
        ),
    ]

    for p in patterns:
        for m in p.finditer(description):
            groups = m.groups()
            # Two patterns have (value, field) order instead of (field, value)
            if groups[0] and re.match(r'^[0-9]', groups[0]):
                val, field = groups[0], groups[1]
            else:
                field, val = groups[0], groups[1]
            field = field.strip().lower()
            val   = val.strip()
            if field not in found:
                found[field] = val

    return [
        (field.capitalize(), val)
        for field, val in found.items()
        if field not in _SKIP_FIELDS
        and len(val.replace(".", "")) <= 7
    ]


def _generate_mrs_from_ui_description(description: str, screen_id: str) -> list[dict]:
    """
    Generate MONOTONICITY and INPUT_TRANSFORMATION MRs directly from
    Agent 1's UI description, used when Agent 2 didn't capture numeric test data.
    These are attached to a synthetic source_tc_id referencing the screen.
    Always confidence="low" since they're not derived from a specific TC.
    """
    mrs = []
    fields = _parse_numeric_fields_from_description(description)

    for field, val in fields:
        try:
            old = float(val)
        except ValueError:
            continue
        if old < 0:
            continue

        new = old + 20
        source_ref = f"UI_DESC_{screen_id}"

        mrs.append(_make_mr(
            mr_id="",
            source_tc_id=source_ref,
            category="INPUT_TRANSFORMATION",
            transformation=f"Change {field} from {int(old)} to {int(new)} (from UI description)",
            follow_up_steps=[
                f"Step 1: Navigate to the screen.",
                f"Step 2: Enter {int(new)} into the {field} field.",
                f"Step 3: Trigger the primary action.",
            ],
            follow_up_test_data=f"{field}: {int(new)}",
            expected_relation="Computation completes successfully with updated output values",
            confidence="low",
        ))

        if old > 0:
            mrs.append(_make_mr(
                mr_id="",
                source_tc_id=source_ref,
                category="MONOTONICITY",
                transformation=f"Increase {field} from {int(old)} to {int(new)} (from UI description)",
                follow_up_steps=[
                    f"Step 1: Navigate to the screen.",
                    f"Step 2: Enter {int(old)} into the {field} field and trigger primary action. Note result.",
                    f"Step 3: Enter {int(new)} into the {field} field and trigger primary action.",
                ],
                follow_up_test_data=f"{field}: {int(new)}",
                expected_relation=f"Result with {field}={int(new)} should be greater than result with {field}={int(old)}",
                confidence="low",
            ))

    return mrs


# ─── Cross-TC INTERACTION_CONSISTENCY ────────────────────────────────────────

def _cross_tc_interaction_consistency(test_cases: list) -> dict | None:
    """
    Detects when 3+ separate TCs each have a single tap on similar elements
    (e.g. 3 separate social media icon TCs) and generates one
    INTERACTION_CONSISTENCY MR that reverses the tap order across them.
    """
    single_tap_tcs = []
    for tc in test_cases:
        if tc.get("type", "").lower() not in ("functional", ""):
            continue
        steps = tc.get("steps", [])
        taps = [s for s in steps if re.search(r"\b(tap|click)\b", s, re.IGNORECASE)]
        if len(taps) == 1:
            single_tap_tcs.append((tc["test_case_id"], taps[0]))

    if len(single_tap_tcs) < 3:
        return None

    theme_groups: dict[str, list] = {}
    for tc_id, step in single_tap_tcs:
        m = re.search(r"\b(icon|tab|button|card)\b", step, re.IGNORECASE)
        if m:
            theme = m.group(1).lower()
            theme_groups.setdefault(theme, []).append((tc_id, step))

    for theme, group in theme_groups.items():
        if len(group) >= 3:
            subset = group[:3]
            source_ids  = [t[0] for t in subset]
            orig_steps  = [t[1] for t in subset]
            rev_steps   = list(reversed(orig_steps))

            clean_steps = [re.sub(r"^Step\s*\d+:\s*", "", s, flags=re.IGNORECASE) for s in rev_steps]
            return _make_mr(
                mr_id="",
                source_tc_id=source_ids[0],
                category="INTERACTION_CONSISTENCY",
                transformation=f"Reverse tap order across {len(subset)} {theme} interactions",
                follow_up_steps=[f"Step {i+1}: {s}" for i, s in enumerate(clean_steps)],
                follow_up_test_data="Same as source tests",
                expected_relation="UI remains stable and no crash or inconsistent state occurs",
                confidence="high",
            )

    return None


# ─── Main entry point ─────────────────────────────────────────────────────────

def generate_metamorphic_relations(tc_data: dict, ui_description: str = "") -> dict:
    """
    Agent 3 (rule-based): derives metamorphic relations from Agent 2 test cases.
    Optionally accepts Agent 1 UI description to extract numeric fields for
    MONOTONICITY and INPUT_TRANSFORMATION when Agent 2 didn't capture them.
    No model or GPU required.
    """
    screen_id = tc_data.get("screen_id", "unknown")
    topic     = tc_data.get("topic", "unknown")
    mrs: list[dict] = []

    for tc in tc_data.get("test_cases", []):
        generated_for_this_tc = 0
        for rule_fn in _RULES:
            if generated_for_this_tc >= 2:   # cap at 2 MRs per source TC
                break
            mr = rule_fn(tc)
            if mr is not None:
                mrs.append(mr)
                generated_for_this_tc += 1

    # ── Cross-TC: INTERACTION_CONSISTENCY from multiple single-tap TCs ────────
    already_has_interaction = any(
        mr["mr_category"] == "INTERACTION_CONSISTENCY" for mr in mrs
    )
    if not already_has_interaction:
        cross_mr = _cross_tc_interaction_consistency(tc_data.get("test_cases", []))
        if cross_mr is not None:
            mrs.append(cross_mr)

    # ── Fallback: use Agent 1 description for numeric MRs if none found from TCs
    if ui_description:
        tc_has_input_transformation = any(
            mr["mr_category"] == "INPUT_TRANSFORMATION" for mr in mrs
        )
        tc_has_monotonicity = any(
            mr["mr_category"] == "MONOTONICITY" for mr in mrs
        )
        if not tc_has_input_transformation or not tc_has_monotonicity:
            desc_mrs = _generate_mrs_from_ui_description(ui_description, screen_id)
            for mr in desc_mrs:
                if mr["mr_category"] == "INPUT_TRANSFORMATION" and not tc_has_input_transformation:
                    mrs.append(mr)
                elif mr["mr_category"] == "MONOTONICITY" and not tc_has_monotonicity:
                    mrs.append(mr)

    # Deduplicate on (source_tc_id, transformation)
    seen: set[str] = set()
    unique_mrs = []
    for mr in mrs:
        key = f"{mr['source_tc_id']}|{mr['transformation']}"
        if key not in seen:
            seen.add(key)
            unique_mrs.append(mr)

    # Sequential IDs
    for i, mr in enumerate(unique_mrs, start=1):
        mr["mr_id"] = f"MR-{i:02d}"

    print(f"   ✅ Agent 3 (rule-based) generated {len(unique_mrs)} metamorphic relations")

    return {
        "screen_id":              screen_id,
        "topic":                  topic,
        "metamorphic_relations":  unique_mrs,
    }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_join(val):
    if not val:
        return ""
    if isinstance(val, str):
        return val
    return ", ".join(
        str(v) if not isinstance(v, dict) else json.dumps(v)
        for v in val
    )


# ─── Save functions ───────────────────────────────────────────────────────────

def save_mr_data(mr_data: dict, out_dir: str = "outputs/metamorphic_relations"):
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