# https://huggingface.co/docs/transformers/v4.41.3/en/training#train-in-native-pytorch

import evaluate
import torch
from datasets import DatasetDict
from peft import LoraConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, get_scheduler


if __name__ == "__main__":
    dataset = DatasetDict.load_from_disk("data/yelp_review_bert")

    train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(dataset["test"], batch_size=8)

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="SEQ_CLS",
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["score"],
    )

    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
    model.add_adapter(config)

    num_epochs = 3
    num_steps = num_epochs * len(train_dataloader)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_steps)

    device = torch.device("cuda:0")
    model.to(device)

    progress_bar = tqdm(range(num_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    metric = evaluate.load("accuracy")

    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    eval_result = metric.compute()
    print(eval_result)

    model.save_pretrained("output/lora/custom_loop")
