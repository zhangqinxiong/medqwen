"""
train.py
Unsloth LoRA fine-tuning of Qwen2-1.5B-Instruct on medical dataset (16-bit mixed precision).
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments


def main():
    model_name = "Qwen/Qwen2-1.5B-Instruct"
    max_seq_length = 1024
    output_dir = "./medical_qwen_lora"
    merged_dir = "./medical_qwen_merged"

    # 1. Load model in 16-bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,       # 16-bit for quality
        dtype=None,                # auto-detect: bfloat16 if supported else float16
    )

    # 2. Configure LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # 3. Load processed data
    print("Loading processed data...")
    train_dataset = load_dataset("json", data_files="train_med.json", split="train")
    eval_dataset = load_dataset("json", data_files="eval_med.json", split="train")

    # Pre-format: apply chat template to messages and add as "text" column
    def format_messages(batch):
        texts = [
            tokenizer.apply_chat_template(msg, tokenize=False)
            for msg in batch["messages"]
        ]
        return {"text": texts}

    train_dataset = train_dataset.map(format_messages, batched=True, remove_columns=["messages"])
    eval_dataset = eval_dataset.map(format_messages, batched=True, remove_columns=["messages"])

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        logging_steps=10,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=2,
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        dataset_text_field="text",
    )

    # 6. Train
    print("Starting training...")
    trainer.train()

    # 7. Save LoRA weights
    print(f"Saving LoRA weights to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 8. Optional: merge LoRA with base model and save full 16-bit model
    print(f"Merging LoRA and saving full model to {merged_dir}")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    print("Training complete!")


if __name__ == "__main__":
    main()
