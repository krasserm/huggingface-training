# https://huggingface.co/docs/transformers/trainer
# https://huggingface.co/docs/peft/accelerate/fsdp

# FSDP config must have mixed precision turned off when using gradient checkpointing, otherwise
# an error is raised, see https://github.com/huggingface/transformers/issues/28499

import evaluate
import numpy as np
import torch
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
)


def main():
    repo_id = "facebook/opt-350m"
    output_path = "output/trainer/sharded/fsdp_lora"

    # no need to add modules_to_save=["classifier"] because
    # get_peft_model() automatically adds it to config
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="SEQ_CLS",
        target_modules=["q_proj", "v_proj"],
    )

    # can also be used with torch_dtype=torch.float16
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, num_labels=5, torch_dtype=torch.float32)
    model = get_peft_model(model, config)

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    metric = evaluate.load("accuracy")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataset = DatasetDict.load_from_disk("data/yelp_review_opt")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=output_path,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_total_limit=2,
        save_strategy="epoch",
        eval_strategy="epoch",
        metric_for_best_model="accuracy",
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],  # .select(range(64)),
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()

    if getattr(trainer.accelerator.state, "fsdp_plugin", None):
        from peft.utils.other import fsdp_auto_wrap_policy

        fsdp_plugin = trainer.accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(output_path)


if __name__ == "__main__":
    main()
