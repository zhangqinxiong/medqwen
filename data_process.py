"""
data_process.py
Download HuatuoGPT dataset, convert to ChatML format, filter by token length, split into train/eval sets.
"""
import os
import json
import random
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

# Use HF mirror for users in China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--num_train", type=int, default=20000)
    parser.add_argument("--num_eval", type=int, default=1000)
    parser.add_argument("--output_train", default="train_med.json")
    parser.add_argument("--output_eval", default="eval_med.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    system_msg = "你是一个专业的临床医生，请根据患者的症状提供准确、严谨且温暖的医学建议。"

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading HuatuoGPT dataset (shibing624/huatuo_medical_qa_sharegpt)...")
    ds = load_dataset("shibing624/huatuo_medical_qa_sharegpt", split="train", streaming=True)

    processed = []
    for i, sample in enumerate(ds):
        conv = sample["conversations"]
        # Build ChatML messages
        messages = [{"role": "system", "content": system_msg}]
        for turn in conv:
            role = "user" if turn["from"] == "human" else "assistant"
            messages.append({"role": role, "content": turn["value"]})

        # Tokenize to check length
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer(text, truncation=False)["input_ids"]

        if len(tokens) <= args.max_seq_length:
            processed.append({"messages": messages})

            if (i + 1) % 50000 == 0:
                print(f"  Processed {i + 1} samples, kept {len(processed)}...")

    print(f"\nTotal processed: {i + 1}, kept (<= {args.max_seq_length} tokens): {len(processed)}")

    random.seed(args.seed)
    random.shuffle(processed)

    train_data = processed[:args.num_train]
    eval_data = processed[args.num_train:args.num_train + args.num_eval]

    print(f"Writing {len(train_data)} training samples to {args.output_train}")
    with open(args.output_train, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    print(f"Writing {len(eval_data)} evaluation samples to {args.output_eval}")
    with open(args.output_eval, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    # Print sample
    print("\n--- Sample entry ---")
    print(json.dumps(train_data[0], ensure_ascii=False, indent=2)[:500])
    print("Done!")


if __name__ == "__main__":
    main()
