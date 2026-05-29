"""
human_eval.py
Generate human evaluation dataset.
Samples 50 diverse questions, runs both models, saves results for expert review.
"""
import os, json, gc, torch, csv, random
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from unsloth import FastLanguageModel

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
MERGED_PATH = "./medical_qwen_merged"
EVAL_FILE = "eval_med.json"
SYSTEM_PROMPT = "你是一个专业的临床医生，请根据患者的症状提供准确、严谨且温暖的医学建议。"

N_SAMPLES = 50
SEED = 42


def load_model(path, label):
    print(f"Loading {label}...")
    m, t = FastLanguageModel.from_pretrained(
        model_name=path, max_seq_length=1024,
        load_in_4bit=False, dtype=None,
    )
    m = FastLanguageModel.for_inference(m)
    return m, t


def generate(model, tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512, temperature=0.7,
            top_p=0.9, repetition_penalty=1.05,
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    random.seed(SEED)

    # Load eval data and sample diverse questions
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Take first 500 and pick every 10th to ensure diversity
    pool = data[:500]
    sampled = [pool[i] for i in range(0, 500, 10)][:N_SAMPLES]
    print(f"Sampled {len(sampled)} questions from {len(pool)} pool")

    questions = []
    ref_answers = []
    for item in sampled:
        msgs = item["messages"]
        q = next((m["content"] for m in msgs if m["role"] == "user"), "")
        a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        questions.append(q)
        ref_answers.append(a)

    # Run FT model
    ft_m, ft_t = load_model(MERGED_PATH, "Fine-tuned")
    ft_outputs = []
    for i, q in enumerate(questions):
        ft_outputs.append(generate(ft_m, ft_t, q))
        if (i + 1) % 10 == 0:
            print(f"  FT [{i+1}/{len(questions)}]")
    del ft_m, ft_t; gc.collect(); torch.cuda.empty_cache()

    # Run Base model
    base_m, base_t = load_model(MODEL_NAME, "Base")
    base_outputs = []
    for i, q in enumerate(questions):
        base_outputs.append(generate(base_m, base_t, q))
        if (i + 1) % 10 == 0:
            print(f"  Base [{i+1}/{len(questions)}]")
    del base_m, base_t; gc.collect(); torch.cuda.empty_cache()

    # ── Save detailed CSV (identifies model) ──
    with open("human_eval_detailed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "model", "output"])
        for i in range(len(questions)):
            w.writerow([i, questions[i], "Base Qwen2-1.5B", base_outputs[i]])
            w.writerow([i, questions[i], "Fine-tuned (Medical)", ft_outputs[i]])

    # ── Save blind evaluation CSV (anonymized A/B) ──
    with open("human_eval_blind.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "question",
            "Model_A_output", "Model_B_output",
            "A_accuracy(1-5)", "A_completeness(1-5)", "A_safety(1-5)",
            "B_accuracy(1-5)", "B_completeness(1-5)", "B_safety(1-5)",
            "preferred(A/B)", "notes"
        ])
        for i in range(len(questions)):
            w.writerow([
                i, questions[i],
                base_outputs[i], ft_outputs[i],
                "", "", "",  # Model A scores
                "", "", "",  # Model B scores
                "",  # Preference
                "",  # Notes
            ])

    # ── Save markdown for easy review ──
    with open("human_eval_preview.md", "w", encoding="utf-8") as f:
        f.write("# Human Evaluation Data\n\n")
        f.write(f"Total samples: {len(questions)}\n")
        f.write(f"Models: Base Qwen2-1.5B (Model A) vs Fine-tuned Medical (Model B)\n\n")
        f.write("---\n\n")
        for i in range(min(10, len(questions))):
            f.write(f"## Sample {i}\n\n")
            f.write(f"**Question:** {questions[i][:200]}\n\n")
            f.write(f"**Model A (Base):** {base_outputs[i][:300]}\n\n")
            f.write(f"**Model B (FT):** {ft_outputs[i][:300]}\n\n")
            f.write(f"**Reference:** {ref_answers[i][:300]}\n\n")
            f.write("---\n\n")

    print(f"\nSaved:")
    print(f"  human_eval_detailed.csv - Full results with model names")
    print(f"  human_eval_blind.csv    - Blind A/B evaluation sheet (fill in scores)")
    print(f"  human_eval_preview.md   - Preview of first 10 samples")
    print("Done!")


if __name__ == "__main__":
    main()
