import json
import re
import csv
import os
import torch
from json_repair import repair_json
from transformers import AutoModelForCausalLM, AutoTokenizer

TESTCASE_PROMPT_TEMPLATE = """
You are a senior QA engineer for mobile apps.
Generate manual test cases from the screen description below.
---
## EXTRACT FIRST
Before writing tests, list:
- INTERACTIVE: buttons, icons, FABs, brand icons (Facebook/LinkedIn/Google+ are ALWAYS interactive — one test each)
- INPUTS: text fields — note any default values visible on screen
- SELECTORS: tabs, toggles, unit switches — list every distinct state and its label
- SCROLLABLE: named scrollable lists — note any visible items by name
- ACCESSIBILITY: font sizes, contrast notes, flagged issues

Do not invent elements. Do not test ads or static labels.

---

## HALLUCINATION GUARD
Before writing each test, ask: "Is this element explicitly mentioned
in the screen description?" If no → do not test it.
Specifically: do NOT invent search fields, calculate buttons, or
tappable list rows unless the description says they exist.

---

## OBSERVATION TESTS ARE BANNED
Never write a test whose only step is "Observe X" or "Look at X".
Every test must have at least one Tap, Enter, Scroll, or Select action.
Static elements (maps, labels, profile cards, placeholder text)
are INFORMATIONAL — do not write tests for them.

---

## SPECIFICITY RULE — CRITICAL
Every test case must use ACTUAL values, labels, and names from the screen description.

WRONG — generic:
- steps: ["Enter a valid age value"]
- expected: ["The value is displayed in the input field"]

RIGHT — screen-specific:
- steps: ["Step 1: Enter '26' into the Age input field"]
- expected: ["'26' remains visible in the Age input field"]

WRONG — generic:
- expected: ["The leaderboard updates"]

RIGHT — screen-specific:
- expected: ["The 'THIS COURSE' tab becomes highlighted and the learner list updates to show only course-specific entries"]

WRONG — generic:
- expected: ["The accessibility requirements are met"]

RIGHT — screen-specific:
- expected: ["Title text (~24sp, bold) and body text (~16sp, regular) have sufficient contrast against the light blue to white gradient background"]

Rules:
- Use exact button labels from the description (e.g. 'CM | KG' not 'unit button')
- Use exact tab names (e.g. 'THIS COURSE' not 'first tab')
- Use exact learner names if visible (e.g. 'Nikolay Nachev' not 'a learner')
- Use exact input field names (e.g. 'Age', 'Waist', 'Height' not 'input field')
- Use exact category names if visible (e.g. 'UnderWeight', 'Normal', 'OverWeight')
- Use exact font sizes and colors from the accessibility section of the description

---

## COVERAGE
- Every INTERACTIVE element → 1 test
- Every INPUT → 2 tests: one valid input with real value, one empty/invalid
- Every SELECTOR STATE → 1 test (e.g. CM|KG and IN|LB = 2 tests)
- Every SCROLLABLE list → 1 test
- Accessibility tests: Generate accessibility test cases covering ALL observations from the description. Each test case must have multiple Verify steps (not just one). Group by category:
  - Font sizes: Verify each text element's sp value
  - Font weight/family: Verify each element's weight and typeface
  - Color contrast: Verify each text/background pair meets WCAG AA (≥4.5:1)
  - Background colors: Verify each surface color
  - Touch targets: Verify each button/tab/FAB/input is at least 44x44px
  - Only generate a category if the description has data for it. type="Accessibility".

---

## WRITING RULES
Steps: use Tap / Enter / Scroll / Select only.

Expected results MUST:
- Name the specific label or element (e.g. "'IN | LB' button becomes active")
- Describe the visible UI change using exact screen values
- Reference real on-screen values (e.g. "rank 1 Nikolay Nachev 56876 XP")
- NEVER say "works correctly", "screen displayed", or "updates successfully"

FAB/CTA: expected result = "A new screen or dialog is presented"
Scroll: name what new content appears (e.g. "entries below rank 5 including ashwinj 5 XP become visible")
Social icons: name the platform (e.g. "LinkedIn app or linkedin.com opens")
Priority: P1 = core action, P2 = navigation/secondary, P3 = edge cases/scroll

---

## OUTPUT
Return ONLY valid JSON, no markdown.

{{
  "module": "",
  "screen_id": "",
  "test_cases": [
    {{
      "test_case_id": "TC_{{screen_id}}_001",
      "title": "",
      "priority": "P1|P2|P3",
      "type": "Functional|Accessibility|Negative",
      "preconditions": [""],
      "test_data": [""],
      "steps": ["Step 1: ..."],
      "expected": [""]
    }}
  ]
}}

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
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    
    print(f"✅ Qwen 7B model loaded")
    return model, tokenizer


def deduplicate_test_cases(test_cases: list) -> list:
    """Remove duplicate test cases based on steps."""
    seen_steps = set()
    unique = []
    for tc in test_cases:
        steps_key = " | ".join(tc.get("steps", []))
        if steps_key not in seen_steps:
            seen_steps.add(steps_key)
            unique.append(tc)
    return unique

def light_post_process(test_cases: list) -> list:
    """
    Light post-processing for Qwen 7B.
    Removes obvious hallucinations.
    """
    
    hallucinated_patterns = [
        "Tap the 'Calculate' button",
        "search field",
    ]
    
    filtered = []
    for tc in test_cases:
        steps_text = " ".join(tc.get("steps", []))
        is_hallucinated = any(pattern in steps_text for pattern in hallucinated_patterns)
        
        if not is_hallucinated:
            filtered.append(tc)
    
    return filtered

def generate_test_cases(ui_data: dict, model, tokenizer) -> dict:
    """Generate test cases using Qwen 7B."""
    prompt = TESTCASE_PROMPT_TEMPLATE.format(
        screen_id=ui_data.get("screen_id", "unknown"),
        topic=ui_data.get("topic", "unknown"),
        description=ui_data.get("description", ""),
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful QA assistant that strictly outputs valid JSON with no markdown."
        },
        {"role": "user", "content": prompt}
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

    print(f"   🔄 Generating test cases (Qwen 7B)...")
    
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=4000,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(
        ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )

    try:
        blocks = re.findall(r'\{.*\}', raw, re.DOTALL)
        if not blocks:
            print(f"   ⚠️  No JSON block found")
            return {"test_cases": []}

        candidate = max(blocks, key=len)
        repaired = repair_json(candidate)
        result = json.loads(repaired)

        result["test_cases"] = deduplicate_test_cases(result.get("test_cases", []))
        result["test_cases"] = light_post_process(result["test_cases"])
        
        print(f"   ✅ Generated {len(result['test_cases'])} test cases")
        return result

    except Exception as e:
        print(f"   ❌ JSON parse failed: {e}")
        return {"test_cases": []}


def _safe_join(val):
    """Join list values with comma separator for CSV."""
    if not val:
        return ""
    # If it's already a string, return as-is
    if isinstance(val, str):
        return val
    # Otherwise join list elements
    return ", ".join(
        str(v) if not isinstance(v, dict) else json.dumps(v)
        for v in val
    )


def save_test_cases(data: dict, out_dir: str = "outputs/testcases"):
    """Save test cases to individual CSV file per screen."""
    os.makedirs(out_dir, exist_ok=True)
    screen_id = data.get("screen_id", "unknown")
    topic = data.get("topic", "unknown")
    module = (data.get("module") or topic or "unknown").strip()
    test_cases = data.get("test_cases", [])

    out_path = os.path.join(out_dir, f"{screen_id}.csv")

    if os.path.exists(out_path):
        os.remove(out_path)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for tc in test_cases:
            w.writerow({
                "test_case_id": tc.get("test_case_id", ""),
                "title": tc.get("title", ""),
                "module": module,
                "screen_id": screen_id,
                "topic": topic,
                "priority": tc.get("priority", ""),
                "type": tc.get("type", ""),
                "preconditions": _safe_join(tc.get("preconditions", [])),
                "test_data": _safe_join(tc.get("test_data", [])),
                "test_steps": _safe_join(tc.get("steps", [])),
                "expected_result": _safe_join(tc.get("expected", [])),
            })

    print(f"  💾 Test cases saved → {out_path}")
    return out_path


def append_to_master_csv(data: dict,
                         master_path: str = "outputs/testcases_master.csv"):
    """Append test cases to master CSV file."""
    os.makedirs(os.path.dirname(master_path), exist_ok=True)
    screen_id = data.get("screen_id", "unknown")
    topic = data.get("topic", "unknown")
    module = (data.get("module") or topic or "unknown").strip()
    test_cases = data.get("test_cases", [])

    exists = os.path.exists(master_path)
    with open(master_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            w.writeheader()
        for tc in test_cases:
            w.writerow({
                "test_case_id": tc.get("test_case_id", ""),
                "title": tc.get("title", ""),
                "module": module,
                "screen_id": screen_id,
                "topic": topic,
                "priority": tc.get("priority", ""),
                "type": tc.get("type", ""),
                "preconditions": _safe_join(tc.get("preconditions", [])),
                "test_data": _safe_join(tc.get("test_data", [])),
                "test_steps": _safe_join(tc.get("steps", [])),
                "expected_result": _safe_join(tc.get("expected", [])),
            })

    print(f"  💾 Appended to master CSV → {master_path}")