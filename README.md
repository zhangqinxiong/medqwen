# MedQwen

Efficient LoRA fine-tuning of Qwen2-1.5B-Instruct for Chinese medical dialogue, based on the Unsloth framework.

## Highlights

- **18.4M trainable parameters** (1.18% of total) — trains on a single **RTX 3060 12GB** in ~5 hours
- **PPL** 9.13 → **5.18** (↓43.2%), **BERTScore F1** 0.6845 → **0.7191** (↑5.0%)
- **72% human preference** over base model in blind A/B evaluation
- Full pipeline: data processing → training → evaluation → Web UI

## Project Structure

| File | Description |
|------|-------------|
| `data_process.py` | Download HuatuoGPT dataset, convert to ChatML, train/eval split |
| `train.py` | LoRA fine-tuning with Unsloth (16-bit mixed precision) |
| `inference.py` | Interactive streaming inference in terminal |
| `app_base.py` | Gradio Web UI — base Qwen2 model |
| `app_ft.py` | Gradio Web UI — fine-tuned MedQwen model |
| `server.py` | FastAPI server serving both models on `/base` and `/medical` |
| `compare.py` | Side-by-side PPL, ROUGE-L, and output comparison |
| `eval_full.py` | Comprehensive evaluation: PPL, ROUGE-L, BLEU, BERTScore |
| `human_eval.py` | Generate paired outputs for blind human evaluation |

## Quick Start

```bash
# 1. Setup environment
bash setup_env.sh
conda activate unsloth_med_qwen

# 2. Prepare data (downloads HuatuoGPT, 20K train + 1K eval)
python data_process.py

# 3. Train (LoRA, ~5 hours on RTX 3060 12GB)
python train.py

# 4. Interactive terminal chat
python inference.py

# 5. Gradio Web UI — fine-tuned model
python app_ft.py

# 6. Compare base vs fine-tuned
python compare.py
# or comprehensive evaluation
python eval_full.py
```

## Training Details

| Config | Value |
|--------|-------|
| Base model | Qwen/Qwen2-1.5B-Instruct |
| LoRA rank | r=16, α=32 |
| Target modules | all attention + FFN projections |
| Trainable params | 18.5M / 1.56B (1.18%) |
| Dataset | HuatuoGPT medical QA (20K training) |
| Precision | BF16 mixed precision |
| Hardware | NVIDIA RTX 3060 12GB |
| Training time | ~5 hours |

## Results

| Metric | Base | MedQwen | Improvement |
|--------|------|---------|-------------|
| Perplexity ↓ | 9.13 | **5.18** | **+43.2%** |
| ROUGE-L ↑ | 0.0940 | **0.1210** | **+28.8%** |
| BLEU ↑ | 0.0010 | **0.0012** | +18.5% |
| BERTScore F1 ↑ | 0.6845 | **0.7191** | **+5.0%** |
| Human preference | 28% | **72%** | — |

## Paper

See the `paper/` directory for LaTeX source — MedQwen: Efficient Low-Resource Fine-Tuning of Qwen2-1.5B for Chinese Medical Question Answering.

## License

Apache 2.0
