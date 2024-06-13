# https://huggingface.co/docs/transformers/training#train-in-native-pytorch
# https://huggingface.co/docs/transformers/accelerate

import evaluate
import torch
from accelerate import Accelerator
from datasets import DatasetDict
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, get_scheduler


if __name__ == "__main__":
    repo_id = "google-bert/bert-base-cased"

    accelerator = Accelerator()
    dataset = DatasetDict.load_from_disk("data/yelp_review_bert")

    train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(dataset["test"], batch_size=8)

    # Demonstrates how to force usage of the PyTorch SDPA (although BERT doesn't suuport anythin else at the moment)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, num_labels=5, attn_implementation="sdpa")
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # prepare data loaders, model, optimizer for distributed training
    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    num_epochs = 3
    num_steps = num_epochs * len(train_dataloader)
    scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_steps)

    progress_bar = tqdm(range(num_steps))
    kernel_options = dict(enable_flash=False, enable_math=True, enable_mem_efficient=False)

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # Force usage of PyTorch SDPA math kernel
            with torch.backends.cuda.sdp_kernel(**kernel_options):
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
        with torch.no_grad(), torch.backends.cuda.sdp_kernel(**kernel_options):
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    # This runs evaluation on all processes with sharded data (= different results)
    # TODO: onvestigate how to gather evaluation results from all processes
    eval_result = metric.compute()
    print(eval_result)

    unwrapped_model = accelerator.unwrap_model(model)

    # Save adapters only
    unwrapped_model.save_pretrained(
        "output/sdpa/math_kernel",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
