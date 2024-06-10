# Training experiments

## Setup

...

```shell
export PYTHONPATH=.
```

## Examples

Documentation links are on top of example scripts.

### Getting started

Demonstrates basic `Trainer` usage.

```shell
python train/basic/getting_started.py
```

This trains the model on all available GPUs using PyTorch DP. To use PyTorch DDP run:

```shell
accelerate launch --config_file accelerate_config.yaml train/basic/getting_started.py
```

### Custom training loop

Demonstrates how to implement a custom training loop. First prepare a dataset:

```shell
python train/custom_loop_dataset.py
```

Run training on single GPU:

```shell
python train/basic/custom_loop.py
python train/lora/custom_loop.py
```

Run training on all GPUs with `accelerate`:

```shell
accelerate launch --config_file accelerate_config.yaml train/basic/custom_loop_accelerate.py
accelerate launch --config_file accelerate_config.yaml train/lora/custom_loop_accelerate.py
```

This uses DDP to train the model (DeepSpeed or FSDP are not enabled in `accelerate_config.yaml`)

## Topics

- ZeRO optimizer
- deepspeed
- fsdp
- flash-atttention-v2

### LoRA and QLoRA

LoRA by default adds adapter for `q_proj` and `v_proj` only, the QLoRA paper (?) adds adapters to all linear layers. This can be done with `target_modules="all-linear"`.

## Resources

- transformers
- accelerate
- peft
- trl

## Articles

- https://www.philschmid.de/fsdp-qlora-llama3
- https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html
