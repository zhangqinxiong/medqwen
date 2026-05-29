"""
inference.py
Interactive streaming inference with fine-tuned Qwen2-1.5B medical model.
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer


def main():
    model_name = "Qwen/Qwen2-1.5B-Instruct"
    lora_path = "./medical_qwen_lora"
    max_seq_length = 1024
    system_prompt = "你是一个专业的临床医生，请根据患者的症状提供准确、严谨且温暖的医学建议。"

    # 1. Load base model + LoRA weights
    print(f"Loading base model: {model_name}")
    print(f"Loading LoRA weights from: {lora_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        dtype=None,
    )
    model = FastLanguageModel.for_inference(model)

    # 2. Streamer for word-by-word output
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 3. Interactive loop
    print("\n=== 医疗 AI 助手（输入 'exit' 退出）===\n")
    while True:
        user_input = input("患者: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda")

        print("医生: ", end="", flush=True)
        _ = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
        )
        print()


if __name__ == "__main__":
    main()
