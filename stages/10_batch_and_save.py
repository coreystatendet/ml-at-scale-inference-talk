import os

# By putting the Huggingface cache on our shared filesystem, we can avoid downloading again in
# future notebooks / experiments.
# By default, the cache lives in the home directory, which will be local to the container.
# This needs to be set before importing transformers!
os.environ["TRANSFORMERS_CACHE"] = "/run/determined/workdir/shared_fs/hfcache"
os.environ["HF_MODULES_CACHE"] = "/run/determined/workdir/shared_fs/hfcache"

import deepspeed
import determined as det
import pandas as pd
import time
from typing import List, Optional
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


hparams = det.get_cluster_info().trial.hparams
tokenizer = transformers.AutoTokenizer.from_pretrained(hparams["model_name"], padding_side="left")
if not (tokenizer.pad_token):
    tokenizer.pad_token = tokenizer.eos_token
with deepspeed.OnDevice(dtype=getattr(torch, hparams["dtype"]), device="meta"):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        hparams["model_name"],
        low_cpu_mem_usage=True,
    )
ds_engine = deepspeed.init_inference(model=model, **hparams["deepspeed_inference_args"])
model = ds_engine.module


def generate(tokenizer, model, batch, max_new_tokens=32) -> List[str]:
    """
    Returns total number of tokens generated.
    """
    with torch.inference_mode():
        tokenize_args = {
            "padding": True,
            "return_tensors": "pt",
            "return_token_type_ids": False,
        }
        # To get prompt lengths for each sample, we need to tokenize independently.
        prompt_lengths = [
            tokenizer(sample, **tokenize_args)["input_ids"].shape[1] for sample in batch
        ]
        batch_tokens = tokenizer(batch, **tokenize_args).to("cuda")
        generations = model.generate(
            **batch_tokens, max_new_tokens=max_new_tokens
        )  # Greedy sampling.
        # Remove the prompt from each generation.
        return [
            tokenizer.decode(generation[prompt_length:], skip_special_tokens=True)
            for generation, prompt_length in zip(generations, prompt_lengths)
        ]


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, lst):
        self.lst = lst

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        return self.lst[idx]


with det.core.init(distributed=det.core.DistributedContext.from_deepspeed()) as core_context:
    # Warm up run; the first generation call is slower due to CUDA initialization.
    generate(tokenizer, model, ["Hello world!"])
    # Splitting up a dataset for distributed training
    dataset = ListDataset(pd.read_csv("dataset.csv")["prompt"].tolist())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=hparams["per_process_batch_size"],
        sampler=torch.utils.data.DistributedSampler(dataset),
    )
    # Timed run.
    prompts = []
    generations = []
    timer = CumulativeTimer()
    timer.start()
    for batch in dataloader:
        prompts += batch
        generations += generate(tokenizer, model, batch)
    timer.stop()
    all_pairs = core_context.distributed.gather(list(zip(prompts, generations)))
    if core_context.distributed.get_rank() == 0:
        # Flatten gathered lists.
        all_pairs = [x for lst in all_pairs for x in lst]
        # Note that the length of the prompt can also affect generation speed, so throughput
        # won't be as consistent as when using a fixed prompt length.
        total_tokens = sum([len(generation) for _, generation in all_pairs])
        throughput = total_tokens / timer.total_time * 1e9
        print(f"Total tokens: {total_tokens}")
        print(f"Throughput: {throughput:.3f} tokens per second")
        core_context.train.report_training_metrics(
            metrics={"throughput": throughput}, steps_completed=1
        )
        with core_context.checkpoint.store_path(metadata={"steps_completed": 1}) as (path, _):
            file_path = os.path.join(path, "generations.csv")
            pd.DataFrame(all_pairs, columns=["prompt", "generation"]).to_csv(file_path)
