"""
stages/text_model_internlm.py
LLM loader for internlm2_5-7b-chat (Shanghai AI Lab / InternLM) — used by
Agents 2, 3, and 4.

IMPORTANT COMPATIBILITY NOTE — read before running the full benchmark:
InternLM's own official examples use a CUSTOM `model.chat(tokenizer, prompt,
history=[])` method, NOT the `tokenizer.apply_chat_template()` +
`model.generate(**inputs)` pattern that stages/test_generation.py,
stages/metamorphic_testing.py, and stages/optimization.py all use for both
your Qwen and Phi setups.

It is NOT guaranteed that internlm2_5-7b-chat's tokenizer has a working
`chat_template` registered for the standard interface — some Hugging Face
repos add one alongside their custom .chat() method, some don't. Rather than
assume either way, warmup() below explicitly runs the EXACT call pattern
your agent stage files use (apply_chat_template -> generate) and will raise
a clear, actionable error if it fails, BEFORE you commit to a long benchmark.

If warmup() fails with something like "chat_template is not set" or produces
garbage/repetitive output, you have two options:
  1. Check if a chat_template exists in tokenizer_config.json on the model's
     Hugging Face page and, if missing, source InternLM's recommended prompt
     format (their <|im_start|>/<|im_end|> convention) and hardcode it as a
     custom chat_template string set on the tokenizer after loading.
  2. Fall back to using model.chat(tokenizer, prompt, history=[]) directly,
     which would require a small adapter in this file that mimics the
     (input_ids -> generate -> decode) shape the agent files expect, OR
     modifying the agent files themselves to branch on model family (a
     larger change — only do this if you're ready to edit that code).

Other differences from the Qwen/Phi loaders:
  - Model ID: internlm/internlm2_5-7b-chat
  - trust_remote_code=True required
  - torch_dtype=torch.bfloat16 (InternLM's own examples use float16; bfloat16
    kept here for consistency with your other loaders — should work fine on
    the same GPU that ran Qwen/Phi in bf16)
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INTERNLM_MODEL_ID = "internlm/internlm2_5-7b-chat"


def load_text_model():
    """
    Load internlm2_5-7b-chat for use as the shared LLM across Agents 2, 3, 4.
    Returns (model, tokenizer) — same signature as the Qwen/Phi loaders so
    the rest of your pipeline works without any other changes, PROVIDED the
    apply_chat_template compatibility check in warmup() passes.
    """
    print(f"   Loading internlm2_5-7b-chat from {INTERNLM_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(
        INTERNLM_MODEL_ID,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        INTERNLM_MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    print("   ✅ internlm2_5-7b-chat loaded")
    return model, tokenizer


def warmup(model, tokenizer):
    """
    Warms up the GPU AND verifies apply_chat_template()+generate() actually
    works for this model/tokenizer combo — the exact pattern your Agent
    2/3/4 stage files use. This is a harder requirement than the Qwen/Phi
    warmups: if this raises or the tokenizer has no chat_template, STOP and
    resolve it (see the module docstring above) before running the full
    benchmark, since every downstream agent call will fail the same way.
    """
    print("   🔥 Warming up internlm2_5-7b-chat...")

    if getattr(tokenizer, "chat_template", None) is None:
        raise RuntimeError(
            "internlm2_5-7b-chat's tokenizer has no chat_template set. "
            "The agent stage files (test_generation.py / metamorphic_testing.py / "
            "optimization.py) require tokenizer.apply_chat_template() to work. "
            "See the compatibility note at the top of stages/text_model_internlm.py "
            "for how to resolve this before running the full benchmark."
        )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hi."},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(
        ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    if not decoded:
        raise RuntimeError(
            "internlm2_5-7b-chat produced an EMPTY response to the warmup "
            "prompt via apply_chat_template()+generate(). The chat_template "
            "may be malformed even though it exists. See the compatibility "
            "note at the top of stages/text_model_internlm.py."
        )

    print(f"   ✅ Warmup complete — sample response: {decoded[:60]!r}")