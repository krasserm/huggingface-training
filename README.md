# Hugging Face training examples

Whirlwind tour through model training with Hugging Face tools based on minimalistic examples. **Work in progress ...**

## Setup

```shell
conda env create -f environment.yml
conda activate huggingface-training

poetry install
invoke precommit-install

export PYTHONPATH=.
```

## Examples

Documentation links are on top of example scripts.

### Trainer class

For basic `Trainer` usage run:

```shell
python scripts/trainer/getting_started.py
```

This trains the model on all available GPUs using PyTorch DP. To use PyTorch DDP run:

```shell
accelerate launch --config_file accelerate_config.yaml scripts/trainer/getting_started.py
```

An example of more advanced `Trainer` usage is:

```shell
accelerate launch --config_file accelerate_config.yaml scripts/trainer/advanced_usage.py
```

A `Trainer` can be customized

- with `TrainingArguments`
- via [subclassing](https://huggingface.co/docs/transformers/v4.41.3/en/trainer#customize-the-trainer)
- via [callbacks](https://huggingface.co/docs/transformers/v4.41.3/en/trainer#callbacks)

### Custom training loop

Demonstrates how to implement a custom training loop. First prepare a dataset:

```shell
python scripts/custom_loop_dataset.py
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

This uses DDP to train the model (DeepSpeed or FSDP are not enabled in `accelerate_config.yaml`)

## Topics

- ZeRO optimizer
- DeepSpeed
- FSDP
- Flash Atttention v2

### LoRA and QLoRA

LoRA reparameterizes a weight matrix W into W = W0 + BA, where W0 is a frozen full-rank matrix and B, A are additive low-rank adapters to be learned. See also [LoRA Recap](https://magazine.sebastianraschka.com/i/141797214/lora-recap) ...

Implementation-wise, LoRA by default adds adapters for `q_proj` and `v_proj` only, the QLoRA paper adds adapters to all linear layers. This can be done with `target_modules="all-linear"` in `LoraConfig`.

### GaLore

[Gradient Low-Rank Projection](https://arxiv.org/abs/2403.03507) (GaLore) is a memory-efficient low-rank training strategy that allows **full-parameter learning** but is more memory-efficient than common low-rank adaptation methods, such as LoRA. GaLore leverages the slow-changing low-rank structure of the gradient G of a weight matrix W, rather than trying to approximate the weight matrix itself as low rank (as in LoRA).

Projection matrices P and Q project gradient matrix G into a low-rank form (P.T)GQ which reduces the memory cost of optimizer states. Because of the slow-changing low-rank structure of G, projection matrices P and Q must only be updated every e.g. 200 iterations which incurs minimal computational overhead.

### NEFTune

[NEFTune](https://arxiv.org/abs/2310.05914) is a technique that can improve performance by adding noise to the embedding vectors during training. It can be enabled by setting `neftune_noise_alpha` in `TrainingArguments`.

## Resources

- transformers
- accelerate
- peft
- trl

## Articles

- https://www.philschmid.de/fsdp-qlora-llama3
- https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html
