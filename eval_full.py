"""
eval_full.py
Comprehensive evaluation: Perplexity, ROUGE-L, BLEU, BERTScore
for Base vs Fine-tuned Qwen2-1.5B on medical dataset.
"""
import os, json, math, gc, torch, sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from unsloth import FastLanguageModel

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
MERGED_PATH = "./medical_qwen_merged"
EVAL_FILE = "eval_med.json"
MAX_SEQ_LEN = 1024
SYSTEM_PROMPT = "你是一个专业的临床医生，请根据患者的症状提供准确、严谨且温暖的医学建议。"

# ── Config ──
N_PPL = 500      # samples for perplexity
N_GEN = 100      # samples for generation-based metrics
DEVICE = "cuda"
OUTPUT_FILE = "eval_results.json"


def load_model(path):
    print(f"  Loading {path}...")
    m, t = FastLanguageModel.from_pretrained(
        model_name=path, max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False, dtype=None,
    )
    m = FastLanguageModel.for_inference(m)
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"  Loaded. GPU mem: {mem:.1f}GB")
    return m, t


def unload_model(m, t):
    del m, t
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Unloaded. GPU mem: {torch.cuda.memory_allocated()/1024**3:.1f}GB")


def compute_ppl(model, tokenizer, dataset, n):
    """Perplexity using sliding window."""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for i, item in enumerate(dataset):
        if i >= n:
            break
        text = tokenizer.apply_chat_template(item["messages"], tokenize=False)
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=MAX_SEQ_LEN).to(DEVICE)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
        total_nll += loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]
        if (i + 1) % 100 == 0:
            print(f"    PPL [{i+1}/{n}]  current_ppl={math.exp(total_nll/total_tokens):.2f}")
    return math.exp(total_nll / total_tokens)


def generate_answer(model, tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7,
                                top_p=0.9, repetition_penalty=1.05)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    # Load dataset
    print(f"\n{'='*60}")
    print("Loading eval dataset...")
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    print(f"  {len(eval_data)} samples loaded")

    results = {}

    # ── Evaluate Fine-tuned ──
    print(f"\n{'='*60}")
    print("EVALUATING: Fine-tuned (Medical)")
    print('='*60)
    ft_m, ft_t = load_model(MERGED_PATH)

    print("  Computing Perplexity...")
    ft_ppl = compute_ppl(ft_m, ft_t, eval_data, N_PPL)
    print(f"  PPL = {ft_ppl:.2f}")
    results["ft_ppl"] = ft_ppl

    print(f"  Generating {N_GEN} answers...")
    ft_answers, ft_refs = [], []
    for i in range(N_GEN):
        item = eval_data[i]
        msgs = item["messages"]
        q = next((m["content"] for m in msgs if m["role"] == "user"), "")
        r = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        a = generate_answer(ft_m, ft_t, q)
        ft_answers.append(a)
        ft_refs.append(r)
        if (i + 1) % 20 == 0:
            print(f"    Generated [{i+1}/{N_GEN}]")
    results["ft_answers"] = ft_answers
    results["ft_refs"] = ft_refs
    unload_model(ft_m, ft_t)

    # ── Evaluate Base ──
    print(f"\n{'='*60}")
    print("EVALUATING: Base Qwen2-1.5B")
    print('='*60)
    base_m, base_t = load_model(MODEL_NAME)

    print("  Computing Perplexity...")
    base_ppl = compute_ppl(base_m, base_t, eval_data, N_PPL)
    print(f"  PPL = {base_ppl:.2f}")
    results["base_ppl"] = base_ppl

    print(f"  Generating {N_GEN} answers...")
    base_answers = []
    for i in range(N_GEN):
        item = eval_data[i]
        q = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
        a = generate_answer(base_m, base_t, q)
        base_answers.append(a)
        if (i + 1) % 20 == 0:
            print(f"    Generated [{i+1}/{N_GEN}]")
    results["base_answers"] = base_answers
    results["base_refs"] = results["ft_refs"]  # same references
    unload_model(base_m, base_t)

    # ── Compute Metrics ──
    print(f"\n{'='*60}")
    print("COMPUTING METRICS")
    print('='*60)
    ft_answers = results["ft_answers"]
    base_answers = results["base_answers"]
    refs = results["ft_refs"]

    # ROUGE-L
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    ft_rouge_f, base_rouge_f = [], []
    for i in range(N_GEN):
        ft_rouge_f.append(scorer.score(refs[i], ft_answers[i])["rougeL"].fmeasure)
        base_rouge_f.append(scorer.score(refs[i], base_answers[i])["rougeL"].fmeasure)
    ft_rouge = sum(ft_rouge_f) / N_GEN
    base_rouge = sum(base_rouge_f) / N_GEN
    results["ft_rougeL"] = ft_rouge
    results["base_rougeL"] = base_rouge
    print(f"  ROUGE-L: Base={base_rouge:.4f}  FT={ft_rouge:.4f}")

    # BLEU
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smooth = SmoothingFunction().method1
    ft_bleu_s, base_bleu_s = [], []
    for i in range(N_GEN):
        ft_bleu_s.append(sentence_bleu([refs[i].split()], ft_answers[i].split(),
                                       smoothing_function=smooth))
        base_bleu_s.append(sentence_bleu([refs[i].split()], base_answers[i].split(),
                                         smoothing_function=smooth))
    ft_bleu = sum(ft_bleu_s) / N_GEN
    base_bleu = sum(base_bleu_s) / N_GEN
    results["ft_bleu"] = ft_bleu
    results["base_bleu"] = base_bleu
    print(f"  BLEU:    Base={base_bleu:.4f}  FT={ft_bleu:.4f}")

    # BERTScore
    from bert_score import score as bert_score
    print("  Computing BERTScore (this may take a few minutes)...")
    for label, answers in [("Base", base_answers), ("FT", ft_answers)]:
        P, R, F1 = bert_score(answers, refs, lang="zh", verbose=False)
        f1_mean = F1.mean().item()
        results[f"{label.lower()}_bertscore"] = f1_mean
        print(f"  BERTScore F1 ({label}): {f1_mean:.4f}")

    # ── Print Summary ──
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print('='*60)
    print(f"{'Metric':<20} {'Base':<12} {'Fine-tuned':<12} {'Change':<12}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    metrics = [
        ("Perplexity ↓", "base_ppl", "ft_ppl", "{:.2f}", "{:+.1f}%", True),
        ("ROUGE-L ↑", "base_rougeL", "ft_rougeL", "{:.4f}", "{:+.1f}%", False),
        ("BLEU ↑", "base_bleu", "ft_bleu", "{:.4f}", "{:+.1f}%", False),
        ("BERTScore F1 ↑", "base_bertscore", "ft_bertscore", "{:.4f}", "{:+.1f}%", False),
    ]
    for name, bk, fk, fmt, chg_fmt, lower_better in metrics:
        bv = results[bk]
        fv = results[fk]
        if lower_better and bv > 0:
            chg = (bv - fv) / bv * 100
        elif not lower_better and bv > 0:
            chg = (fv - bv) / bv * 100
        else:
            chg = 0
        print(f"{name:<20} {fmt.format(bv):<12} {fmt.format(fv):<12} {chg_fmt.format(chg):<12}")

    # Save
    save = {k: v for k, v in results.items() if not isinstance(v, list)}
    with open(OUTPUT_FILE, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")
    print("Done!")


if __name__ == "__main__":
    main()
