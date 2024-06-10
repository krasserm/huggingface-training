from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset = load_dataset("yelp_review_full")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(4000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    small_datasets = DatasetDict({"train": small_train_dataset, "test": small_eval_dataset})
    small_datasets.save_to_disk("dataset/yelp_review_small")
