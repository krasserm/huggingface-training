# https://huggingface.co/docs/transformers/training#train-in-native-pytorch

import evaluate
import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig


if __name__ == "__main__":
    small_datasets = DatasetDict.load_from_disk("data/yelp_review_opt")
    eval_dataloader = DataLoader(small_datasets["test"], batch_size=8)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "output/trainer/sharded/fsdp_qlora",
        num_labels=5,
        quantization_config=bnb_config,
        device_map="cuda:0",
    )

    metric = evaluate.load("accuracy")
    device = torch.device("cuda:0")

    model.eval()

    for batch in tqdm(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    eval_result = metric.compute()
    print(eval_result)
