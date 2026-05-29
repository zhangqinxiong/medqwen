"""
compare.py
Compare base Qwen2-1.5B-Instruct vs fine-tuned medical model.
Metrics: Perplexity (PPL), ROUGE-L, side-by-side output comparison.
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import math
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
LORA_PATH = "./medical_qwen_lora"
EVAL_FILE = "eval_med.json"
MAX_SEQ_LEN = 1024
SYSTEM_PROMPT = "你是一个专业的临床医生，请根据患者的症状提供准确、严谨且温暖的医学建议。"

TEST_QUESTIONS = [
    "我最近头痛、发烧、喉咙痛，已经三天了，请问可能是什么病？",
    "我膝盖上长了一块小红疙瘩，边缘泛红，不疼不痒，是什么问题？",
    "我最近总是失眠，晚上躺下要一两个小时才能睡着，该怎么办？",
    "我体检发现血压偏高，140/90，平时需要注意什么？",
    "我吃东西胃酸反流，烧心，吃什么药比较好？",
]


def load_model(lora=False):
    print(f"\nLoading model: {MODEL_NAME} {'+ LoRA' if lora else '(base only)'}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        dtype=None,
    )
    if lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, LORA_PATH)
    model = FastLanguageModel.for_inference(model)
    return model, tokenizer


def compute_perplexity(model, tokenizer, dataset, max_samples=100):
    """Compute perplexity on a dataset."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    count = 0

    for i, item in enumerate(dataset):
        if i >= max_samples:
            break
        messages = item["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
        input_ids = inputs["input_ids"].to("cuda")
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            nll = loss.item() * input_ids.shape[1]
            total_nll += nll
            total_tokens += input_ids.shape[1]
            count += 1
        if (i + 1) % 50 == 0:
            print(f"  Perplexity calc: {i + 1}/{max_samples}")

    ppl = math.exp(total_nll / total_tokens)
    return ppl


def generate_answer(model, tokenizer, question):
    """Generate a single answer."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
    )
    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return answer.strip()


def compute_rougel(reference, hypothesis):
    """Compute ROUGE-L score."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"]


def main():
    print("=" * 60)
    print("Base Model vs Fine-tuned Model Comparison")
    print("=" * 60)

    # Load eval dataset
    print("\nLoading eval dataset...")
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    print(f"Loaded {len(eval_data)} eval samples")

    # Load models
    base_model, base_tokenizer = load_model(lora=False)
    ft_model, ft_tokenizer = load_model(lora=True)

    # ── Part 1: Qualitative comparison ──
    print("\n" + "=" * 60)
    print("PART 1: Side-by-side Answer Comparison")
    print("=" * 60)

    for i, question in enumerate(TEST_QUESTIONS):
        print(f"\n--- Question {i + 1}: {question} ---")

        base_answer = generate_answer(base_model, base_tokenizer, question)
        ft_answer = generate_answer(ft_model, ft_tokenizer, question)

        print(f"\n  [Base Model]:")
        print(f"    {base_answer[:300]}")
        print(f"\n  [Fine-tuned]:")
        print(f"    {ft_answer[:300]}")
        print()

    # ── Part 2: Perplexity ──
    print("=" * 60)
    print("PART 2: Perplexity (PPL) on Eval Set")
    print("(lower = better, measures how well the model predicts medical responses)")
    print("=" * 60)

    base_ppl = compute_perplexity(base_model, base_tokenizer, eval_data, max_samples=100)
    ft_ppl = compute_perplexity(ft_model, ft_tokenizer, eval_data, max_samples=100)

    print(f"\n  Base Model PPL:     {base_ppl:.2f}")
    print(f"  Fine-tuned PPL:     {ft_ppl:.2f}")
    print(f"  Improvement:        {(base_ppl - ft_ppl) / base_ppl * 100:.1f}%" if ft_ppl < base_ppl else "  No improvement")

    # ── Part 3: ROUGE-L ──
    print("\n" + "=" * 60)
    print("PART 3: ROUGE-L Score on Eval Set")
    print("(higher = better, measures answer overlap with doctor's reference)")
    print("=" * 60)

    base_rouge = {"precision": 0, "recall": 0, "fmeasure": 0}
    ft_rouge = {"precision": 0, "recall": 0, "fmeasure": 0}
    n_eval = min(50, len(eval_data))

    for i in range(n_eval):
        item = eval_data[i]
        # Get the user question and assistant answer from messages
        user_msg = ""
        assistant_msg = ""
        for m in item["messages"]:
            if m["role"] == "user":
                user_msg = m["content"]
            elif m["role"] == "assistant":
                assistant_msg = m["content"]

        reference = assistant_msg

        base_answer = generate_answer(base_model, base_tokenizer, user_msg)
        ft_answer = generate_answer(ft_model, ft_tokenizer, user_msg)

        try:
            bs = compute_rougel(reference, base_answer)
            fs = compute_rougel(reference, ft_answer)
            base_rouge["precision"] += bs.precision
            base_rouge["recall"] += bs.recall
            base_rouge["fmeasure"] += bs.fmeasure
            ft_rouge["precision"] += fs.precision
            ft_rouge["recall"] += fs.recall
            ft_rouge["fmeasure"] += fs.fmeasure
        except Exception:
            pass

        if (i + 1) % 10 == 0:
            print(f"  ROUGE-L calc: {i + 1}/{n_eval}")

    for name, scores in [("Base Model", base_rouge), ("Fine-tuned", ft_rouge)]:
        print(f"\n  {name} ROUGE-L:")
        print(f"    Precision: {scores['precision'] / n_eval:.4f}")
        print(f"    Recall:    {scores['recall'] / n_eval:.4f}")
        print(f"    F1:        {scores['fmeasure'] / n_eval:.4f}")

    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
