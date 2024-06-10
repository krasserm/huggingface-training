from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


def create_splits():
    dataset = load_dataset("yelp_review_full")

    train_dataset = dataset["train"].shuffle(seed=42).select(range(4000))
    eval_dataset = dataset["test"].shuffle(seed=42).select(range(1000))

    return train_dataset, eval_dataset


def tokenize_split(tokenizer, split):
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_split = split.map(tokenize_function, batched=True)
    tokenized_split = tokenized_split.remove_columns(["text"])
    tokenized_split = tokenized_split.rename_column("label", "labels")
    tokenized_split.set_format("torch")

    return tokenized_split


def tokenize_splits(tokenizer, train_dataset, eval_dataset):
    return DatasetDict(
        {
            "train": tokenize_split(tokenizer, train_dataset),
            "test": tokenize_split(tokenizer, eval_dataset),
        }
    )


def main():
    tokenizer_1 = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    tokenizer_2 = AutoTokenizer.from_pretrained("facebook/opt-125m")
    tokenizer_2.model_max_length = 512

    train_dataset, eval_dataset = create_splits()

    tokenize_splits(tokenizer_1, train_dataset, eval_dataset).save_to_disk("data/yelp_review_bert")
    tokenize_splits(tokenizer_2, train_dataset, eval_dataset).save_to_disk("data/yelp_review_opt")


if __name__ == "__main__":
    main()
