# Training experiments

## Setup

...

```shell
export PYTHONPATH=.
```

## Examples

Documentation links are on top of example script.

### Getting started

Demonstrates basic `Trainer` usage.

```shell
python train/getting_started.py
```

Trains model on all available GPUs by default. It looks like DDP is not used here and model is sharded layer-wise.

### Custom training loop

Demonstrates how to implement a custom training loop. First prepare a dataset:

```shell
python train/custom_loop_dataset.py
```

Run training on single GPU:

```shell
python train/custom_loop.py
```

Run training all GPUs with `accelerate`:

```shell
accelerate launch --config_file accelerate_config.yaml train/custom_loop_accelerate.py
```

This uses DDP to train the model (DeepSpeed or FSDP are not enabled in `accelerate_config.yaml`)

## Topics

- ZeRO optimizer
- deepspeed
- fsdp
- flash-atttention-v2

## Resources

- transformers
- accelerate
- peft
- trl

## Articles

- https://www.philschmid.de/fsdp-qlora-llama3
- https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html
