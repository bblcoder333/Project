import json
import re
import csv
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ─── PROMPT ───────────────────────────────────────────────────────────────────
TESTCASE_PROMPT_TEMPLATE = """
You are a senior QA engineer (mobile apps). Generate REALISTIC, high-quality test cases from the given screen description.

Input is a SCREEN DESCRIPTION extracted from a mobile screenshot (components + actions + navigation + buttons).
Your goal: produce a set of test cases that a senior tester would actually execute.

Hard rules:
- Do NOT invent UI elements that are not mentioned.
- Do NOT reference "the image" or "the screenshot"; reference only what is described.
- Prefer deterministic, step-by-step actions.
- Cover: positive flows, negative flows, boundary/validation, toggles state, clickable buttons, checkboxes, select, input, navigation, accessibility/usability, interruption/resume, and data persistence where applicable.
- If a test requires unknown app behavior, write assumption in Preconditions (short).

Return ONLY valid JSON (no markdown, no commentary).

JSON schema:
{{
  "module": "<short module name>",
  "screen_id": "<screen id>",
  "test_cases": [
    {{
      "test_case_id": "TC_<screen_id>_001",
      "title": "<clear title>",
      "priority": "P0|P1|P2",
      "type": "Functional|Negative|UI|Accessibility|Usability|Regression",
      "preconditions": ["..."],
      "test_data": ["..."],
      "steps": ["Step 1 ...", "Step 2 ...", "Step 3 ..."],
      "expected": ["..."]
    }}
  ]
}}

Screen ID: {screen_id}
Topic: {topic}

RETRIEVED KNOWLEDGE (use this to inform your test cases):
{retrieved_context}

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


# ─── INFERENCE ────────────────────────────────────────────────────────────────
def generate_test_cases(ui_data: dict, retrieved_chunks: list,
                        model, tokenizer) -> dict:
    context = "\n".join([f"- {c['text']}" for c in retrieved_chunks])

    prompt = TESTCASE_PROMPT_TEMPLATE.format(
        screen_id        = ui_data.get("screen_id", "unknown"),
        topic            = ui_data.get("topic", "unknown"),
        retrieved_context= context,
        description      = ui_data.get("description", ""),
        elements         = ui_data.get("structured_elements", ""),
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=3500,
    ).to("cuda")

    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=800,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(
        ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )

    try:
        blocks = re.findall(r'\{.*\}', raw, re.DOTALL)
        if not blocks:
            raise ValueError("No JSON block found in output")
        candidate = max(blocks, key=len)
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse failed: {e}")
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