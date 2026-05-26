import json
import re
import csv
import os
import torch
from json_repair import repair_json
from transformers import AutoModelForCausalLM, AutoTokenizer


TESTCASE_PROMPT_TEMPLATE = """
You are a senior QA engineer specializing in mobile applications.

Generate ONLY high-value manual test cases from the given mobile screen description.

The input is a SCREEN DESCRIPTION extracted from a screenshot. It contains UI elements and accessibility observations.

Your task is to generate realistic behavioral test cases based ONLY on explicitly described UI elements.

---

## STEP 1 — SCREEN SCAN (UI ONLY)

Extract ONLY visible and interactive UI elements.

INTERACTIVE_ELEMENTS:
- Each clickable element explicitly named (buttons, icons, FABs)

NAVIGATIONAL_ELEMENTS:
- menu icon, back icon (ONLY if explicitly mentioned)

SCROLLABLE_ELEMENTS:
- lists or scrollable sections (ONLY if explicitly described as list/scrollable)

INFORMATIONAL_ELEMENTS:
- labels, cards, images, maps, static text, ads (NOT interactive)

INPUT_FIELDS:
- each input field explicitly named (age, waist, height, etc.)

SELECTORS:
- tabs, toggles, dropdowns, unit/gender selectors

STRICT RULES:
- DO NOT group elements (each icon/button is separate)
- DO NOT invent UI elements
- DO NOT treat informational elements as interactive
- ADS MUST NEVER be tested

The SCREEN SCAN is STRICTLY BINDING.
Every element listed MUST appear in at least one test case.

---

## STEP 1.5 — ACCESSIBILITY OBSERVATIONS (NOT UI ELEMENTS)

Extract accessibility-related information separately:

ACCESSIBILITY_OBSERVATIONS:
- font size notes
- color contrast notes
- touch target size notes
- readability notes
- icon distinguishability notes

IMPORTANT RULES:
- Accessibility observations are NOT UI elements
- They MUST NOT be tapped, scrolled, or interacted with
- They are ONLY used to create accessibility test cases

---

## STEP 2 — COVERAGE REQUIREMENT (MANDATORY)

You MUST ensure full coverage:

- Every INTERACTIVE element → at least 1 test case
- Every INPUT_FIELD → at least 1 test case
- Every SELECTOR → at least 1 test case
- Every NAVIGATIONAL element → at least 1 test case
- Every SCROLLABLE element → at least 1 test case (if explicitly scrollable)
- Accessibility_observations → MUST produce exactly 1 accessibility test case

DO NOT stop until full coverage is achieved.

---

## STEP 3 — TEST GENERATION RULES

Generate test cases by iterating through each scanned element.

Rules:
- One UI element → at least one test case
- Do NOT skip any UI element
- Do NOT assume hidden functionality
- Do NOT generate tests for informational elements
- Always include EXACTLY ONE accessibility test derived from ACCESSIBILITY_OBSERVATIONS

Allowed actions:
- Tap
- Enter
- Select
- Scroll
- Swipe

---

## FAB RULE

If a Floating Action Button exists:

- Only describe immediate visible UI change
- Do NOT assume backend or data persistence

GOOD:
"A new screen or dialog for adding a learner is presented"

BAD:
"New learner is successfully added"

---

## EXPECTED RESULT RULES

Each expected result must describe:
- what changed
- which UI element changed

BAD:
- "Screen displayed"
- "Works correctly"
- "Opened successfully"

GOOD:
- "The TOTAL tab becomes highlighted and leaderboard updates"
- "Entered value remains visible in the waist input field"

---

## PRIORITY RULES

- P1 → primary CTA / core action
- P2 → navigation / important interaction
- P3 → secondary interaction
- P4 → accessibility test

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

---

## FINAL CHECK (MANDATORY)

Before returning:

- Every UI element has at least one test case
- Accessibility observations are used in exactly one accessibility test
- No hallucinated UI elements exist
- No informational elements are tested interactively
- Output is complete and not truncated

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