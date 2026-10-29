"""
stages/text_model_mistral.py
LLM loader for Mistral-7B-Instruct-v0.3 (Mistral AI) — used by Agents 2, 3, and 4.

This file only contains the model loading and warmup functions.
All generate_* functions (generate_test_cases, generate_metamorphic_relations,
optimize_metamorphic_relations) are unchanged and imported directly from their
existing stage files — only the model/tokenizer loaded here is swapped in.

Notes:
  - Model ID: mistralai/Mistral-7B-Instruct-v0.3 (Apache 2.0)
  - Gated on Hugging Face: open the model page once while logged in and accept
    the contact-info prompt (instant), then run `huggingface-cli login` (or set
    HF_TOKEN) on the machine that runs the benchmark.
  - No trust_remote_code needed (native transformers support).
  - torch_dtype=torch.bfloat16 (~14.5 GB; fits a 24 GB GPU)
  - tokenizer.pad_token_id must be set to eos_token_id explicitly.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MISTRAL_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"


def load_text_model():
    """
    Load Mistral-7B-Instruct-v0.3 for use as the shared LLM across Agents 2, 3, 4.
    Returns (model, tokenizer) — same signature as the Qwen/Phi versions so the
    rest of your pipeline works without any other changes.
    """
    print(f"   Loading Mistral-7B-Instruct-v0.3 from {MISTRAL_MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL_ID)
    # Mistral doesn't set a pad token by default.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MISTRAL_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print("   ✅ Mistral-7B-Instruct-v0.3 loaded")
    return model, tokenizer


def warmup(model, tokenizer):
    """
    Single throwaway inference pass to warm up GPU caches before timed runs.
    Uses a minimal prompt so it completes near-instantly.
    """
    print("   🔥 Warming up Mistral-7B-Instruct-v0.3...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hi."},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        raise RuntimeError(
            "Mistral's chat template rejected a system message "
            f"({e}). Merge the system prompt into the first user message in "
            "your stage files."
        ) from e

    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("   ✅ Warmup complete")