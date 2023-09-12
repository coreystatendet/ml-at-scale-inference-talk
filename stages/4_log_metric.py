import os

# By putting the Huggingface cache on our shared filesystem, we can avoid downloading again in
# future notebooks / experiments.
# By default, the cache lives in the home directory, which will be local to the container.
# This needs to be set before importing transformers!
os.environ["TRANSFORMERS_CACHE"] = "/run/determined/workdir/shared_fs/hfcache"
os.environ["HF_MODULES_CACHE"] = "/run/determined/workdir/shared_fs/hfcache"

import determined as det
import time
from typing import Optional
import torch
import transformers


class CumulativeTimer:
    def __init__(self) -> None:
        self.total_time = 0
        self.last_started: Optional[int] = None

    def start(self) -> None:
        if self.last_started:
            raise RuntimeError("Timer already started")
        self.last_started = time.monotonic_ns()

    def stop(self) -> None:
        if not self.last_started:
            raise RuntimeError("Timer not started")
        self.total_time += time.monotonic_ns() - self.last_started
        self.last_started = None


MODEL_NAME = "gpt2"

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
if not (tokenizer.pad_token):
    tokenizer.pad_token = tokenizer.eos_token
model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda")


def generate(tokenizer, model, batch_size=1, max_new_tokens=32) -> int:
    """
    Returns total number of tokens generated.
    """
    with torch.inference_mode():
        batch = [tokenizer.bos_token] * batch_size
        batch_tokens = tokenizer(
            batch,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
            add_special_tokens=False,
        ).to("cuda")
        generations = model.generate(
            **batch_tokens, max_new_tokens=max_new_tokens
        )  # Greedy sampling.
        # The return value includes the initial BOS token; we remove to avoid counting it.
        return sum(len(generation[1:]) for generation in generations)


with det.core.init() as core_context:
    # Warm up run; the first generation call is slower due to CUDA initialization.
    generate(tokenizer, model, batch_size=1)

    # Timed run.
    timer = CumulativeTimer()
    timer.start()
    total_tokens = generate(tokenizer, model, batch_size=1)
    timer.stop()
    throughput = total_tokens / timer.total_time * 1e9
    print(f"Total tokens: {total_tokens}")
    print(f"Throughput: {throughput:.3f} tokens per second")
    core_context.train.report_training_metrics(
        metrics={"throughput": throughput}, steps_completed=1
    )
