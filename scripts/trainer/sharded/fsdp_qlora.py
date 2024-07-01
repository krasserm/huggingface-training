# https://huggingface.co/docs/transformers/trainer
# https://huggingface.co/docs/peft/accelerate/fsdp#use-peft-qlora-and-fsdp-for-finetuning-large-models-on-multiple-gpus

import evaluate
import numpy as np
import torch
from datasets import DatasetDict
from peft import LoraConfig
from peft.utils.other import fsdp_auto_wrap_policy
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorWithPadding,
)
from trl import get_kbit_device_map, SFTTrainer


def main():
    repo_id = "facebook/opt-350m"
    output_path = "output/trainer/sharded/fsdp_qlora"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    # no need to add modules_to_save=["classifier"] because
    # get_peft_model() automatically adds it to config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="SEQ_CLS",
        target_modules=["q_proj", "v_proj"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        repo_id,
        num_labels=5,
        device_map=get_kbit_device_map(),
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )

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
        metric_for_best_model="accuracy",
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=lora_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    with training_args.main_process_first():
        trainer.model.print_trainable_parameters()

    trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    trainer.save_model(output_path)


if __name__ == "__main__":
    main()
