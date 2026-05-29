# Evaluation Results: Base vs Fine-tuned Qwen2-1.5B

**Date:** 2026-05-28
**Dataset:** HuatuoGPT medical QA (eval_med.json, 1000 samples)
**Model:** Qwen/Qwen2-1.5B-Instruct + LoRA (r=16, merged 16-bit)
**Metrics:** Perplexity (N=500), ROUGE-L/BLEU/BERTScore (N=100)

---

## Summary Table

| Metric | Base | Fine-tuned | Change | Direction |
|--------|------|------------|--------|-----------|
| Perplexity | 9.13 | **5.18** | +43.2% | ↓ better |
| ROUGE-L | 0.0940 | **0.1210** | +28.8% | ↑ better |
| BLEU | 0.0010 | **0.0012** | +18.5% | ↑ better |
| BERTScore F1 | 0.6845 | **0.7191** | +5.0% | ↑ better |

## Raw Values

```json
{
  "base_ppl": 9.13,
  "ft_ppl": 5.18,
  "base_rougeL": 0.0940,
  "ft_rougeL": 0.1210,
  "base_bleu": 0.0010,
  "ft_bleu": 0.0012,
  "base_bertscore": 0.6845,
  "ft_bertscore": 0.7191
}
```

## LaTeX Table

```latex
\begin{table}[h]
\centering
\caption{Evaluation results on HuatuoGPT medical test set.}
\label{tab:eval}
\begin{tabular}{lccc}
\toprule
Metric & Base & Fine-tuned & $\Delta$ \\
\midrule
Perplexity $\downarrow$ & 9.13 & 5.18 & +43.2\% \\
ROUGE-L $\uparrow$ & 0.0940 & 0.1210 & +28.8\% \\
BLEU $\uparrow$ & 0.0010 & 0.0012 & +18.5\% \\
BERTScore F1 $\uparrow$ & 0.6845 & 0.7191 & +5.0\% \\
\bottomrule
\end{tabular}
\end{table}
```

## Notes

- Perplexity measures prediction confidence on held-out medical dialogues
- ROUGE-L uses Chinese text without stemming
- BLEU uses smoothing function method1 for zero-count n-grams
- BERTScore uses default Chinese BERT model (bert-base-chinese)
- All generation uses: temperature=0.7, top_p=0.9, repetition_penalty=1.05, max_new_tokens=256
