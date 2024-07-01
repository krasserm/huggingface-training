# Hugging Face training examples

Whirlwind tour through model training with Hugging Face tools based on minimalistic examples. **Work in progress ...**

## Setup

```shell
conda env create -f environment.yml
conda activate huggingface-training

poetry install
invoke precommit-install

# https://github.com/Dao-AILab/flash-attention
pip install flash-attn --no-build-isolation

export PYTHONPATH=.
```

## Examples

Documentation links are at the top of example scripts.

### Trainer class

Basic `Trainer` usage:

```shell
python scripts/trainer/getting_started.py
```

This trains the model on all available GPUs using PyTorch DP. To use PyTorch DDP run:

```shell
accelerate launch --config_file accelerate_config.yaml scripts/trainer/getting_started.py
```

A more advanced `Trainer` usage example is:

```shell
accelerate launch --config_file accelerate_config.yaml scripts/trainer/advanced_usage.py
```

A `Trainer` can be customized

- with `TrainingArguments`
- via [subclassing](https://huggingface.co/docs/transformers/trainer#customize-the-trainer)
- via [callbacks](https://huggingface.co/docs/transformers/en/trainer#callbacks)

### Custom training loop

Demonstrates how to implement a custom training loop. First prepare a dataset:

```shell
python scripts/prepare_datasets.py
```

Run training on single GPU:

```shell
python scripts/basic/custom_loop.py
python scripts/lora/custom_loop.py
```

Run training on all GPUs with `accelerate`:

```shell
accelerate launch --config_file accelerate_config.yaml scripts/basic/custom_loop_accelerate.py
accelerate launch --config_file accelerate_config.yaml scripts/lora/custom_loop_accelerate.py
```

This uses DDP to train the model.

### Attention implementations

Examples how to enforce usage of different scaled dot product attention (SDPA) implementations. The following example enforces usage of the [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) implementation:

```shell
accelerate launch --config_file accelerate_config.yaml scripts/sdpa/flash_attention.py
```

This one enforces usage of the PyTorch SDPA "math" implementation:

```shell
accelerate launch --config_file accelerate_config.yaml scripts/sdpa/math_kernel.py
```

See [this section](https://pytorch.org/blog/out-of-the-box-acceleration/#flash-attention-memory-efficient-attention--math-differences) for an overview of PyTorch SDPA implementations.

### Sharding strategies

FSDP training example:

```shell
accelerate launch --config_file accelerate_config_fsdp.yaml scripts/trainer/sharded/fsdp.py
```

FSDP with LoRA training example:

```shell
accelerate launch --config_file accelerate_config_fsdp_lora.yaml scripts/trainer/sharded/fsdp_lora.py
```

FSDP with QLoRA training example:

```shell
accelerate launch --config_file accelerate_config_fsdp_qlora.yaml scripts/trainer/sharded/fsdp_qlora.py
```

The next example is a DeepSpeed ZeRO 3 training example. DeepSpeed can either be configured with an `accelerate` config file via the [Accelerate DeepSpeed Plugin](https://huggingface.co/docs/accelerate/usage_guides/deepspeed#accelerate-deepspeed-plugin) or a DeepSpeed [configuration file](deepspeed_config.json) referenced by `TrainingArguments` in [zero.py](scripts/trainer/sharded/zero.py). A DeepSpeed configuration file provides more flexibility and [configuration options](https://www.deepspeed.ai/docs/config-json/). Training can either be launched with `accelerate`

```shell
accelerate launch --config_file accelerate_config.yaml scripts/trainer/sharded/zero.py
```

or with the `deepspeed` launcher:

```shell
deepspeed scripts/trainer/sharded/zero.py
```

## Topics

### FSDP

[Fully Sharded Data Parallel](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) (FSDP) is a data parallel method that shards a model’s parameters, gradients and optimizer states across the number of available GPUs. It is very similar to [DeepSpeed ZeRO](https://deepspeed.readthedocs.io/en/latest/zero3.html) stage 3. See also [Fully Sharded Data Parallelism](https://blog.clika.io/fsdp-1/) for an introduction to FSDP and [Efficiently fine-tune Llama 3 with PyTorch FSDP and Q-Lora](https://www.philschmid.de/fsdp-qlora-llama3) for combining FSDP with QLoRA.

### DeepSpeed

From Hugging Face [DeepSpeed](https://huggingface.co/docs/transformers/main/en/main_classes/deepspeed) documentation:

> [DeepSpeed](https://github.com/microsoft/DeepSpeed), powered by [Zero Redundancy Optimizer](https://deepspeed.readthedocs.io/en/latest/zero3.html) (ZeRO), is an optimization library for training and fitting very large models onto a GPU. It is available in several ZeRO stages, where each stage progressively saves more GPU memory by partitioning the optimizer state, gradients, parameters, and enabling offloading to a CPU or NVMe.

- [DeepSpeed with Trainer](https://huggingface.co/docs/transformers/main/en/deepspeed)
- [DeepSpeed with Accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed)

### Flash Attention

From [attention optimizations](https://huggingface.co/docs/transformers/llm_optims#attention-optimizations):

> FlashAttention and FlashAttention-v2 break up the attention computation into smaller chunks and reduce the number of intermediate read/write operations to GPU memory to speed up inference (and training). FlashAttention-2 improves on the original FlashAttention algorithm by also parallelizing over sequence length dimension and better partitioning work on the hardware to reduce synchronization and communication overhead.

From [GPU inference](https://huggingface.co/docs/transformers/perf_infer_gpu_one?install=NVIDIA#expected-speedups):

> FlashAttention-2 does not support computing attention scores with padding tokens. You must manually pad/unpad the attention scores for batched inference when the sequence contains padding tokens. This leads to a significant slowdown for batched generations with padding tokens. To overcome this, you should use FlashAttention-2 without padding tokens in the sequence during training (by packing a dataset or concatenating sequences until reaching the maximum sequence length).

PyTorch SDPA with Flash Attention v2 is used by default with `torch>=2.1.1`

### LoRA and QLoRA

LoRA reparameterizes a weight matrix W into W = W0 + BA, where W0 is a frozen full-rank matrix and B, A are additive low-rank adapters to be learned. See also [LoRA Recap](https://magazine.sebastianraschka.com/i/141797214/lora-recap) ...

Implementation-wise, LoRA by default adds adapters for `q_proj` and `v_proj` only, the QLoRA paper adds adapters to all linear layers. This can be done with `target_modules="all-linear"` in `LoraConfig`.

### GaLore

[Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507) (GaLore) is a memory-efficient low-rank training strategy that allows **full-parameter learning** but is more memory-efficient than common low-rank adaptation methods, such as LoRA. GaLore leverages the slow-changing low-rank structure of the gradient G of a weight matrix W, rather than trying to approximate the weight matrix itself as low rank (as in LoRA).

Projection matrices P and Q project gradient matrix G into a low-rank form (P.T)GQ which reduces the memory cost of optimizer states. Because of the slow-changing low-rank structure of G, projection matrices P and Q must only be updated every e.g. 200 iterations which incurs minimal computational overhead.

GaLore can be configured in `TrainingArguments` as described [here](https://huggingface.co/docs/transformers/trainer#galore).

### NEFTune

[NEFTune](https://arxiv.org/abs/2310.05914) is a technique that can improve performance by adding noise to the embedding vectors during training. It can be enabled by setting `neftune_noise_alpha` in `TrainingArguments`.
