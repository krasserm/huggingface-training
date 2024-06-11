# https://huggingface.co/docs/transformers/quicktour#trainer---a-pytorch-optimized-training-loop
# https://huggingface.co/docs/transformers/v4.41.3/en/training#train-with-pytorch-trainer

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
    Trainer,
)


if __name__ == "__main__":
    repo_id = "distilbert/distilbert-base-uncased"
    output_path = "output/trainer/getting_started"

    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    def tokenize_dataset(batch):
        return tokenizer(batch["text"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataset = load_dataset("rotten_tomatoes")
    dataset = dataset.map(tokenize_dataset, batched=True)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=output_path,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        save_total_limit=2,
        save_strategy="epoch",
        eval_strategy="epoch",
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

    # uses all GPUs by default
    trainer.train()
    trainer.save_model(output_path)
