"""
stages/text_model_llama3.py
LLM loader for Meta-Llama-3-8B-Instruct (Meta) — used by Agents 2, 3, and 4.

This file only contains the model loading and warmup functions.
All generate_* functions (generate_test_cases, generate_metamorphic_relations,
optimize_metamorphic_relations) are unchanged and imported directly from their
existing stage files — only the model/tokenizer loaded here is swapped in.

Notes:
  - Model ID: meta-llama/Meta-Llama-3-8B-Instruct (Llama 3 Community License)
  - Gated on Hugging Face: open the model page once while logged in and
    request access / accept Meta's license (approval is manual, so it may not
    be instant), then run `huggingface-cli login` (or set HF_TOKEN) on the
    machine that runs the benchmark. Without access, from_pretrained() fails
    with a 401/403 — that is not a code bug.
  - No trust_remote_code needed (native transformers support, >= 4.40).
  - torch_dtype=torch.bfloat16 (~16 GB; fits a 24 GB GPU)
  - tokenizer.pad_token_id must be set to eos_token_id explicitly.
  - STOP TOKENS: Llama-3's chat template ends a turn with <|eot_id|>, which is
    a different id from the tokenizer's eos_token (<|end_of_text|>). The stage
    files call generate() with pad_token_id=tokenizer.eos_token_id and do not
    override eos_token_id, so they fall back to model.generation_config.
    load_text_model() sets that to BOTH ids, so every stage file stops cleanly
    without being edited.
  - CONTEXT WINDOW: Llama-3-8B has an 8192-token window for input + output
    combined. The Agent 3/4 stage files allow up to 8192 input tokens plus
    thousands of new tokens, so a very long prompt on a busy screen can
    overrun it. If that shows up, change LLAMA3_MODEL_ID to
    "meta-llama/Llama-3.1-8B-Instruct" (128K context, same loader works).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LLAMA3_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def load_text_model():
    """
    Load Meta-Llama-3-8B-Instruct for use as the shared LLM across Agents 2, 3, 4.
    Returns (model, tokenizer) — same signature as the Qwen/Phi/Mistral versions
    so the rest of your pipeline works without any other changes.
    """
    print(f"   Loading Meta-Llama-3-8B-Instruct from {LLAMA3_MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(LLAMA3_MODEL_ID)
    # Llama-3 doesn't set a pad token by default.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA3_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Make generate() stop on <|eot_id|> as well as <|end_of_text|>, for every
    # call in every stage file (see STOP TOKENS note above).
    stop_ids = [tokenizer.eos_token_id]
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot_id is not None and eot_id != tokenizer.unk_token_id and eot_id not in stop_ids:
        stop_ids.append(eot_id)
    model.generation_config.eos_token_id = stop_ids

    print("   ✅ Meta-Llama-3-8B-Instruct loaded")
    return model, tokenizer


def warmup(model, tokenizer):
    """
    Single throwaway inference pass to warm up GPU caches before timed runs.
    Calls generate() the same way the stage files do (no eos_token_id override),
    and raises if the model doesn't stop on a trivial prompt — that would mean
    the stop-token setup in load_text_model() isn't taking effect, and every
    agent call would ramble until max_new_tokens.
    """
    print("   🔥 Warming up Meta-Llama-3-8B-Instruct...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hi in one short sentence."},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    inputs.pop("token_type_ids", None)

    max_new = 64
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = ids[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    if not decoded:
        raise RuntimeError(
            "Meta-Llama-3-8B-Instruct produced an EMPTY response to the warmup "
            "prompt. Check that the model downloaded correctly and that your "
            "Hugging Face account has been granted access to it."
        )
    if len(new_ids) >= max_new:
        raise RuntimeError(
            "Meta-Llama-3-8B-Instruct did not stop on its end-of-turn token "
            f"for a trivial prompt (used all {max_new} new tokens). Check "
            "model.generation_config.eos_token_id includes <|eot_id|>."
        )

    print(f"   ✅ Warmup complete — sample response: {decoded[:60]!r}")