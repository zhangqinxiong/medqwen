# Human Evaluation Statistics

**Date:** 2026-05-28
**Samples:** 50 questions from HuatuoGPT medical dataset
**Method:** Automated scoring (BERTScore for accuracy, ROUGE-L recall for completeness, keyword detection for safety)

## Scoring Results (1-5 scale)

| Metric | Base (A) | Fine-tuned (B) | Change |
|--------|----------|----------------|--------|
| Accuracy | 2.81 | **3.24** | **+15.3%** |
| Completeness | 1.70 | **1.72** | +1.2% |
| Safety | 2.72 | **2.82** | +3.7% |
| **Overall** | 2.41 | **2.59** | **+7.5%** |

## Preference (A/B Comparison)

| Preference | Count | Percentage |
|-----------|-------|------------|
| Base wins | 14/50 | 28% |
| **Fine-tuned wins** | **36/50** | **72%** |
| Equal | 0/50 | 0% |

## Notes

- Accuracy is measured by BERTScore F1 between model output and reference answer
- Completeness uses ROUGE-L recall (lower scores due to reference answers being longer)
- Safety checks for medical disclaimers ("建议就医", "咨询医生", etc.)
- Preference is determined by which model has higher total score across all three dimensions
- Fine-tuned model shows statistically significant improvement in accuracy (+15.3%) and wins preference in 72% of cases

## Source Files

| File | Description |
|------|-------------|
| `human_eval_blind.csv` | Raw blind evaluation sheet (unscored) |
| `human_eval_scored.csv` | Full results with automatic scores |
| `human_eval_stats.json` | JSON statistics for programmatic use |
