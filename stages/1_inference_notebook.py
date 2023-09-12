!pip install transformers

import os
# By putting the Huggingface cache on our shared filesystem, we can avoid downloading again in
# future notebooks / experiments.
# By default, the cache lives in the home directory, which will be local to the container.
# This needs to be set before importing transformers!
os.environ["TRANSFORMERS_CACHE"] = "/run/determined/workdir/shared_fs/hfcache"
os.environ["HF_MODULES_CACHE"] = "/run/determined/workdir/shared_fs/hfcache"

import torch
import transformers

MODEL_NAME = "gpt2"

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
if not(tokenizer.pad_token):
    tokenizer.pad_token = tokenizer.eos_token
model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda")


def generate(tokenizer, model, batch_size=1, max_new_tokens=32):
    with torch.inference_mode():
        batch = [tokenizer.bos_token] * batch_size
        batch_tokens = tokenizer(
            batch,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
            add_special_tokens=False,
        ).to("cuda")
        generations = model.generate(**batch_tokens, max_new_tokens=max_new_tokens)  # Greedy sampling.
        # skip_special_tokens = True removes the BOS, EOS tokens.
        texts = [
            tokenizer.decode(generation, skip_special_tokens=True) for generation in generations
        ]
        for i, text in enumerate(texts):
            print(f"Text {i+1}:\n{text}\n\n")



