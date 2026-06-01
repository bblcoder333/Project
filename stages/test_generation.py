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

## RULE 1: EXTRACT ONLY WHAT EXISTS
Before writing tests, extract ONLY these elements explicitly shown in the description:
- INTERACTIVE: buttons, icons, FABs, tabs, toggles
- INPUTS: text fields
- SCROLLABLE: lists or scrollable sections
- ACCESSIBILITY: font sizes, contrast notes

Do NOT invent elements. Do NOT test static labels, ads, or images.

---

## RULE 2: HALLUCINATION GUARD
Before writing ANY test:
1. Point to the exact sentence in the description that mentions this element
2. If you cannot find it → DO NOT WRITE A TEST FOR IT

Forbidden without explicit mention:
- Calculate buttons
- Search fields
- Social media icons (unless shown)
- Input fields (unless shown)
- Unit toggles (unless shown)

If it is not in the description → it does not exist.

---

## RULE 3: NO OBSERVATION-ONLY TESTS
Every test MUST have at least one action: Tap, Enter, Scroll, or Select.

FORBIDDEN:
- "Step 1: Observe the button"
- "Step 1: Verify the text is visible"

REQUIRED:
- "Step 1: Tap the button"
- "Step 1: Enter '25' into the field"

Exception: Accessibility tests may use "Verify" steps.

---

## RULE 4: COVERAGE
- Every INTERACTIVE element → 1 test minimum
- Every INPUT field → 2 tests (valid input + empty/invalid)
- Every TAB or TOGGLE → 1 test per state
- Every SCROLLABLE section → 1 test
- Exactly 1 accessibility test (not multiple)

---

## RULE 5: SPECIFIC EXPECTED RESULTS
Use exact labels, values, and names from the description.

WRONG: "The button works"
RIGHT: "The 'Get Started' button navigates to a new screen"

WRONG: "The field accepts input"
RIGHT: "'25' remains visible in the Age field"

WRONG: "More items appear"
RIGHT: "Entries below rank 5 become visible"

---

## RULE 6: SOCIAL ICON RULE (IF PRESENT)
Each social media icon = ONE separate test case.
Never combine into one test.
Facebook → "The Facebook app or facebook.com opens"
LinkedIn → "The LinkedIn app or linkedin.com opens"
Google+ → "The Google+ app or google.com/+ opens"

---

## RULE 7: TOGGLE RULE (IF PRESENT)
A toggle button is ONE action, not multiple.
FORBIDDEN: "Step 1: Tap CM|KG. Step 2: Tap CM"
REQUIRED: "Step 1: Tap the 'CM | KG' button"

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