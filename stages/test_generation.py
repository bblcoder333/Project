import json
import re
import csv
import os
import torch
from json_repair import repair_json
from transformers import AutoModelForCausalLM, AutoTokenizer

TESTCASE_PROMPT_TEMPLATE = """
You are a senior QA engineer for mobile apps.

Generate high-value manual test cases ONLY from explicitly described UI elements.

The input is a screen description containing UI elements and accessibility observations.

---

## STEP 1 — EXTRACT UI ELEMENTS

Extract only explicitly mentioned items:

INTERACTIVE_ELEMENTS: clickable UI (buttons, icons, FAB)
NAVIGATIONAL_ELEMENTS: menu/back icons
INPUT_FIELDS: text inputs (age, waist, height, etc.)
SELECTORS: tabs, toggles, dropdowns, unit/gender selectors
SCROLLABLE_ELEMENTS: lists or scrollable sections (only if stated)
INFORMATIONAL_ELEMENTS: labels, cards, maps, static text, ads (not interactive)

Rules:
- Do NOT invent elements
- Do NOT group elements
- Only listed items are valid test targets
- Ads must NOT be tested

## ICON DETECTION RULE
- Brand/social icons (Facebook, LinkedIn, Google+, Twitter etc.) are ALWAYS interactive
- Profile cards with a name and image are ALWAYS tappable
- Maps with location pins are ALWAYS tappable
- These must appear in INTERACTIVE_ELEMENTS even if not explicitly labeled "clickable"

---

## STEP 2 — ACCESSIBILITY

ACCESSIBILITY_OBSERVATIONS include:
- font size
- contrast
- readability
- touch target size

Rules:
- Not interactive UI
- Used only for ONE accessibility test case

---

## STEP 3 — COVERAGE RULES

Generate test cases ensuring full coverage:

- Each INTERACTIVE element → at least 1 test
- Each INPUT_FIELD → at least 1 test
- Each SELECTOR → at least 1 test
- Each NAVIGATIONAL element → at least 1 test
- Each SCROLLABLE element → at least 1 test
- Exactly 1 accessibility test from observations

Do not skip any element.

---

## STEP 4 — TEST RULES

Allowed actions:
Tap, Enter, Select, Scroll, Swipe

Rules:
- One element = one test minimum
- No backend assumptions
- No hidden functionality assumptions
- Do not test informational elements

FAB rule:
- Only describe immediate UI change after tap
- Expected result: "A new screen or dialog is presented"
- Do NOT say "new entry is successfully added"

TOGGLE / UNIT SWITCH RULE:
- A unit switch is ONE button that alternates between states
- Test as: Step 1: Tap the unit switch button. Step 2: Observe the units change.
- Do NOT assume two separate buttons exist
- Do NOT write two steps for a single toggle

---

## EXPECTED RESULT RULES

Must describe observable UI change only.

Bad:
- "Works correctly"
- "Screen displayed"
- "All learners are visible as the user scrolls"

Good:
- "TOTAL tab becomes highlighted and leaderboard updates to show overall rankings"
- "Entered value remains visible in the Age input field"
- "Additional learner entries below the currently visible list become visible"

For scroll tests:
- Describe what NEW content appears not just that scrolling works
- Good: "Additional learner entries below rank 5 become visible"
- Bad: "All learners are visible as the user scrolls"

---

## PRIORITY RULES

- P1 → primary CTA / core action
- P2 → navigation / important interaction / accessibility
- P3 → secondary interactions

---

## OUTPUT FORMAT

Return ONLY valid JSON. No markdown. No commentary.

{{
  "module": "<module name>",
  "screen_id": "<screen id>",
  "element_scan": {{
    "interactive_elements": [],
    "navigational_elements": [],
    "scrollable_elements": [],
    "informational_elements": [],
    "input_fields": [],
    "selectors": [],
    "accessibility_observations": []
  }},
  "test_cases": [
    {{
      "test_case_id": "TC_{screen_id}_001",
      "title": "",
      "priority": "P1",
      "type": "Functional|Accessibility|Negative",
      "preconditions": [""],
      "test_data": [""],
      "steps": [
        "Step 1: ..."
      ],
      "expected": [
        "Specific observable UI change"
      ]
    }}
  ]
}}

Ensure:
- No hallucinated UI elements
- Full coverage of extracted elements
- Exactly one accessibility test
- No informational element interactions
- No duplicate test cases

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


def load_text_model():
    model_id  = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    print(f"✅ Text model loaded — {model_id}")
    return model, tokenizer


def deduplicate_test_cases(test_cases: list) -> list:
    seen_steps = set()
    unique     = []
    for tc in test_cases:
        steps_key = " | ".join(tc.get("steps", []))
        if steps_key not in seen_steps:
            seen_steps.add(steps_key)
            unique.append(tc)
    return unique


def generate_test_cases(ui_data: dict, model, tokenizer) -> dict:
    prompt = TESTCASE_PROMPT_TEMPLATE.format(
        screen_id   = ui_data.get("screen_id", "unknown"),
        topic       = ui_data.get("topic", "unknown"),
        description = ui_data.get("description", ""),
    )

    messages = [
        {"role": "system", "content": "You are a helpful QA assistant that strictly outputs valid JSON."},
        {"role": "user",   "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(
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
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(
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