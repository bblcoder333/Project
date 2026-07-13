"""
stages/text_model_phi.py
LLM loader for Phi-3.5-mini-instruct (Microsoft) — used by Agents 2, 3, and 4.

This file only contains the model loading and warmup functions.
All generate_* functions (generate_test_cases, generate_metamorphic_relations,
optimize_metamorphic_relations) are unchanged and imported directly from their
existing stage files — only the model/tokenizer loaded here is swapped in.

Key differences from the Qwen2.5-7B-Instruct loader:
  - Model ID: microsoft/Phi-3.5-mini-instruct
  - trust_remote_code=True required
  - torch_dtype=torch.bfloat16 (explicit, not "auto")
  - _attn_implementation="eager" for broad compatibility
  - AutoTokenizer instead of AutoTokenizer (same class, different defaults)
  - tokenizer.pad_token_id must be set to eos_token_id explicitly
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHI_MINI_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

def load_text_model():
    """
    Load Phi-3.5-mini-instruct for use as the shared LLM across Agents 2, 3, 4.
    Returns (model, tokenizer) — same signature as the Qwen version so the
    rest of your pipeline works without any other changes.
    """
    print(f"   Loading Phi-3.5-mini-instruct from {PHI_MINI_MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(
        PHI_MINI_MODEL_ID,
        trust_remote_code=True,
    )
    # Phi-3.5-mini doesn't set pad_token by default — must match eos_token
    # to avoid warnings and ensure correct generation stopping behavior.
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        PHI_MINI_MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",  # use 'flash_attention_2' if available
    )

    print("   ✅ Phi-3.5-mini-instruct loaded")
    return model, tokenizer


def warmup(model, tokenizer):
    """
    Single throwaway inference pass to warm up GPU caches before timed runs.
    Uses a minimal prompt so it completes near-instantly.
    """
    print("   🔥 Warming up Phi-3.5-mini-instruct...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hi."},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("   ✅ Warmup complete")