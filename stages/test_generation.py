import json
import re
import csv
import os
import torch
from json_repair import repair_json


TESTCASE_PROMPT_TEMPLATE = """
You are a senior QA engineer specializing in mobile applications.

Generate ONLY high-value manual test cases from the given mobile screen description.

The input is a SCREEN DESCRIPTION extracted from a screenshot. It may contain UI elements such as buttons, text, cards, lists, tabs, fields, and labels. It also contains visual design information including font sizes, colors, contrast, and accessibility observations.

Your task is to generate realistic behavioral test cases a QA engineer would execute.

Before writing any test case, extract ONLY what is explicitly mentioned:

INTERACTIVE_ELEMENTS: [list every button, tab, icon, input field, FAB, CTA found]
ICONS_DETECTED: [menu icon? yes/no] [info icon? yes/no] [back arrow? yes/no]
LISTS_DETECTED: [yes/no — name them]
INPUT_FIELDS: [list each one by name]
SELECTORS: [gender selector? unit toggle? yes/no]

Do NOT add any element not in the description.

## STEP 2 — GENERATE TEST CASES

Rules (strict):
- Only test elements listed in your STEP 1 scan
- If menu icon detected → generate 1 test (tap opens drawer)
- If info icon detected → generate 1 test (tap opens dialog)
- If input fields detected → generate 1 valid input test
- If tabs detected → generate 1 tab switch test
- If list detected → generate 1 scroll test
- If gender/unit selector detected → generate 1 selection test per selector
- Always generate exactly 1 Accessibility test checking: font size ≥12sp, color contrast, touch targets
- Generate 5–8 total test cases
- Every test must have a specific observable expected result (name the element + the change)
- Steps contain only: Tap / Enter / Scroll / Swipe / Select

FORBIDDEN:
- Do not reference any button, field, or behavior not in your STEP 1 scan
- Do not say an element is "displayed" or "works correctly" in expected results
- Do not assume non-interactive elements are tappable

## EXAMPLE

Screen has: tabs (THIS COURSE, TOTAL), list (TOP LEARNERS), FAB, menu icon.

Scan:
INTERACTIVE_ELEMENTS: [THIS COURSE tab, TOTAL tab, TOP LEARNERS list, FAB, menu icon]
ICONS_DETECTED: [menu icon: yes] [info icon: no] [back arrow: no]
LISTS_DETECTED: [yes — TOP LEARNERS]
INPUT_FIELDS: [none]
SELECTORS: [none]

Good test:
{{
  "title": "Switch between THIS COURSE and TOTAL tabs",
  "steps": ["Step 1: Tap THIS COURSE tab", "Step 2: Tap TOTAL tab"],
  "expected": ["Leaderboard list updates to show total rankings across all courses"]
}}

Bad test (DO NOT do this):
{{
  "title": "Sort leaderboard by XP",
  "steps": ["Tap sort icon"],
  "expected": ["List sorted by XP"]
}}
Reason: No sort icon was in the scan.

## OUTPUT FORMAT

Return ONLY valid JSON, no markdown:

{{
  "module": "<module name>",
  "screen_id": "<screen id>",
  "element_scan": {{
    "interactive_elements": [],
    "icons_detected": {{}},
    "lists": [],
    "input_fields": [],
    "selectors": []
  }},
  "test_cases": [
    {{
      "test_case_id": "TC_{screen_id}_001",
      "title": "",
      "priority": "P1",
      "type": "Functional|Accessibility|Negative",
      "preconditions": [""],
      "test_data": [""],
      "steps": ["Step 1: ...", "Step 2: ..."],
      "expected": ["Specific observable outcome"]
    }}
  ]
}}

## RECHECK

After completing the scan, re-read each test case and verify:
"Is every element in this test listed in my INTERACTIVE_ELEMENTS scan?"
If no → delete the test case before outputting.

Screen ID: {screen_id}
Topic: {topic}

SCREEN DESCRIPTION:
\"\"\"
{description}
\"\"\"
"""

CSV_COLUMNS = [
    "test_case_id",
    "title",
    "module",
    "screen_id",
    "topic",
    "priority",
    "type",
    "preconditions",
    "test_data",
    "test_steps",
    "expected_result",
]


def deduplicate_test_cases(test_cases: list) -> list:
    seen_steps = set()
    unique     = []
    for tc in test_cases:
        steps_key = " | ".join(tc.get("steps", []))
        if steps_key not in seen_steps:
            seen_steps.add(steps_key)
            unique.append(tc)
    return unique


def generate_test_cases(ui_data: dict, model, processor) -> dict:
    prompt = TESTCASE_PROMPT_TEMPLATE.format(
        screen_id   = ui_data.get("screen_id", "unknown"),
        topic       = ui_data.get("topic", "unknown"),
        description = ui_data.get("description", ""),
    )

    messages = [
        {"role": "system", "content": "You are a helpful QA assistant that strictly outputs valid JSON."},
        {"role": "user",   "content": prompt}
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to("cuda")

    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=3000,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    raw = processor.tokenizer.decode(
        ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )

    try:
        blocks = re.findall(r'\{.*\}', raw, re.DOTALL)
        if not blocks:
            print(f"⚠️  No JSON block found. Raw output preview:\n{raw[:300]}")
            return {"test_cases": []}

        candidate = max(blocks, key=len)
        repaired  = repair_json(candidate)
        result    = json.loads(repaired)

        # Remove duplicate test cases before saving
        result["test_cases"] = deduplicate_test_cases(result.get("test_cases", []))
        return result

    except Exception as e:
        print(f"❌ JSON parse failed: {e}")
        print(f"⚠️  Raw output preview:\n{raw[:300]}")
        return {"test_cases": []}


def _safe_join(val):
    if not val:
        return ""
    return " | ".join(
        str(v) if not isinstance(v, dict) else json.dumps(v)
        for v in val
    )


def save_test_cases(data: dict, out_dir: str = "outputs/testcases"):
    os.makedirs(out_dir, exist_ok=True)
    screen_id  = data.get("screen_id", "unknown")
    topic      = data.get("topic", "unknown")
    module     = (data.get("module") or topic or "unknown").strip()
    test_cases = data.get("test_cases", [])

    out_path = os.path.join(out_dir, f"{screen_id}.csv")

    # Delete existing file first so it's always a fresh write
    if os.path.exists(out_path):
        os.remove(out_path)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for tc in test_cases:
            w.writerow({
                "test_case_id":    tc.get("test_case_id", ""),
                "title":           tc.get("title", ""),
                "module":          module,
                "screen_id":       screen_id,
                "topic":           topic,
                "priority":        tc.get("priority", ""),
                "type":            tc.get("type", ""),
                "preconditions":   _safe_join(tc.get("preconditions", [])),
                "test_data":       _safe_join(tc.get("test_data", [])),
                "test_steps":      _safe_join(tc.get("steps", [])),
                "expected_result": _safe_join(tc.get("expected", [])),
            })

    print(f"  💾 Test cases saved → {out_path}")
    return out_path


def append_to_master_csv(data: dict,
                         master_path: str = "outputs/testcases_master.csv"):
    os.makedirs(os.path.dirname(master_path), exist_ok=True)
    screen_id  = data.get("screen_id", "unknown")
    topic      = data.get("topic", "unknown")
    module     = (data.get("module") or topic or "unknown").strip()
    test_cases = data.get("test_cases", [])

    exists = os.path.exists(master_path)
    with open(master_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            w.writeheader()
        for tc in test_cases:
            w.writerow({
                "test_case_id":    tc.get("test_case_id", ""),
                "title":           tc.get("title", ""),
                "module":          module,
                "screen_id":       screen_id,
                "topic":           topic,
                "priority":        tc.get("priority", ""),
                "type":            tc.get("type", ""),
                "preconditions":   _safe_join(tc.get("preconditions", [])),
                "test_data":       _safe_join(tc.get("test_data", [])),
                "test_steps":      _safe_join(tc.get("steps", [])),
                "expected_result": _safe_join(tc.get("expected", [])),
            })

    print(f"  💾 Appended to master CSV → {master_path}")