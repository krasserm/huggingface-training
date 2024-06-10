# https://huggingface.co/docs/transformers/v4.41.3/en/training#train-in-native-pytorch
# https://huggingface.co/docs/transformers/v4.41.3/en/accelerate

import evaluate
import torch
from accelerate import Accelerator
from datasets import DatasetDict
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, get_scheduler


if __name__ == "__main__":
    accelerator = Accelerator()

    small_datasets = DatasetDict.load_from_disk("dataset/yelp_review_small")
    small_train_dataset = small_datasets["train"]
    small_eval_dataset = small_datasets["test"]

    train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)

    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # prepare data loaders, model, optimizer for distributed training
    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    num_epochs = 3
    num_steps = num_epochs * len(train_dataloader)
    scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_steps)

    progress_bar = tqdm(range(num_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    metric = evaluate.load("accuracy")

    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    # This runs evaluation on all processes with sharded data (= different results)
    # TODO: onvestigate how to gather evaluation results from all processes
    eval_result = metric.compute()
    print(eval_result)
