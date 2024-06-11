# https://huggingface.co/docs/transformers/v4.41.3/en/trainer

import evaluate
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
)

from scripts.prepare_datasets import create_splits


def prepare_dataset(tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    train_dataset, eval_dataset = create_splits()
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    return train_dataset, eval_dataset


def main():
    repo_id = "google-bert/bert-base-cased"
    output_path = "output/trainer/advanced_usage"

    model = AutoModelForSequenceClassification.from_pretrained(repo_id, num_labels=5)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    metric = evaluate.load("accuracy")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset, eval_dataset = prepare_dataset(tokenizer)

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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        # trainer doesn't evalauate by default,
        # this is required for custom metrics
        compute_metrics=compute_metrics,
    )

    # train from scratch or resume from latest checkpoint
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model(output_path)


if __name__ == "__main__":
    main()
