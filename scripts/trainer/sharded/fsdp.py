# https://huggingface.co/docs/transformers/trainer

import evaluate
import numpy as np
from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
)


def main():
    repo_id = "facebook/opt-350m"
    output_path = "output/trainer/sharded/fsdp"

    model = AutoModelForSequenceClassification.from_pretrained(repo_id, num_labels=5)
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
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        dataloader_num_workers=2,  # this defaults to 0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        # trainer doesn't evalauate by default,
        # this is required for custom metrics
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(output_path)


if __name__ == "__main__":
    main()
