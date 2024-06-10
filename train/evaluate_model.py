# https://huggingface.co/docs/transformers/v4.41.3/en/training#train-in-native-pytorch

import evaluate
import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification


if __name__ == "__main__":
    small_datasets = DatasetDict.load_from_disk("data/yelp_review_bert")
    eval_dataloader = DataLoader(small_datasets["test"], batch_size=8)

    model = AutoModelForSequenceClassification.from_pretrained("output/basic/custom_loop_accelerate", num_labels=5)

    metric = evaluate.load("accuracy")
    device = torch.device("cuda:0")

    model.to(device)
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
