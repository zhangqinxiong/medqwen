# Paper Outline: Medical Qwen — Efficient Fine-tuning of Qwen2-1.5B for Chinese Medical Dialogue

## Target Venues
- **ACL Rolling Review / ACL Short Paper** (4-6 pages)
- **EMNLP Short Paper** (4-6 pages)
- **COLING Short Paper** (4-6 pages)
- **AAAI / IJCAI Deployed Application Track** (6-8 pages)

---

## 1. Title and Abstract

### Title
**MedQwen: Efficient Low-Resource Fine-Tuning of Qwen2-1.5B for Chinese Medical Question Answering**

### Abstract (150-200 words)
- **Background:** Large language models (LLMs) show promise in clinical NLP but require domain adaptation
- **Problem:** Full fine-tuning is computationally prohibitive; medical domain needs specialized knowledge
- **Method:** Apply LoRA (Low-Rank Adaptation) to Qwen2-1.5B-Instruct using the Unsloth framework, trained on 20K HuatuoGPT Chinese medical conversation pairs
- **Results:** Perplexity reduced from 9.13 to 5.18 (43.2%↓); BERTScore improves by 5.0%; human preference 72% over base model
- **Conclusion:** Lightweight fine-tuning (18.4M trainable parameters, 1.18% of total) effectively adapts general-purpose LLMs to the medical domain with minimal computational cost (~5 hours on a single RTX 3060)

---

## 2. Introduction (1-1.5 pages)

### 2.1 Motivation
- Growing demand for intelligent medical consultation systems
- General-purpose LLMs lack specialized medical knowledge
- Resource constraints: most hospitals/researchers cannot afford large-scale training

### 2.2 Challenges
- Medical domain requires high accuracy and safety
- Full fine-tuning of LLMs is GPU-intensive
- Chinese medical data has unique linguistic and clinical characteristics

### 2.3 Contributions
- Demonstrate that LoRA fine-tuning with only 18.4M parameters achieves significant medical domain adaptation
- Comprehensive evaluation framework: PPL + ROUGE-L + BLEU + BERTScore + Human Evaluation
- Open-source pipeline: data processing, training, inference, and web deployment
- Single-GPU feasibility: entire pipeline runs on RTX 3060 (12GB) in under 5 hours

### 2.4 Paper Organization
- Brief roadmap of remaining sections

---

## 3. Related Work (0.5-1 page)

### 3.1 Medical LLMs
- Overview of medical LLMs: Med-PaLM 2, GPT-4 for clinical applications
- Chinese medical LLMs: BianQue, BenTsao (Huatuo), Zhongjing,神农
- Gap: most require substantial computational resources

### 3.2 Parameter-Efficient Fine-Tuning
- Adapter-based methods: LoRA (Hu et al., 2022), AdaLoRA
- Prefix tuning, prompt tuning
- Advantages for resource-constrained scenarios

### 3.3 Medical QA Evaluation
- Traditional metrics: BLEU, ROUGE (limitations for dialogue)
- Semantic metrics: BERTScore
- Human evaluation protocols in clinical NLP

---

## 4. Methodology (1-1.5 pages)

### 4.1 Base Model
- **Qwen2-1.5B-Instruct:** architecture overview (transformer decoder, 1.5B parameters)
- Pre-training data and capabilities
- Why 1.5B: balance between performance and resource requirements

### 4.2 LoRA Fine-Tuning

#### 4.2.1 LoRA Formulation
- Mathematical formulation: $h = W_0 x + \Delta W x = W_0 x + BAx$
- Rank decomposition: $r = 16$, $\alpha = 32$, dropout = 0

#### 4.2.2 Target Modules
- All attention projections: q_proj, k_proj, v_proj, o_proj
- All feed-forward projections: gate_proj, up_proj, down_proj
- Total trainable parameters: 18,464,768 / 1,562,179,072 (1.18%)

#### 4.2.3 Training Configuration
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 2 × 10⁻⁴ |
| Scheduler | Cosine (warmup ratio 0.1) |
| Batch Size | 4 (per device), accum 4 → effective 16 |
| Epochs | 3 |
| Precision | BF16 mixed precision |
| Gradient Checkpointing | Enabled |
| Training Time | ~5 hours |
| Hardware | NVIDIA RTX 3060 (12GB) |

### 4.3 Training Framework: Unsloth
- Kernel optimizations: improved attention, reduced memory footprint
- Compatibility with HuggingFace ecosystem

### 4.4 Dataset: HuatuoGPT

#### 4.4.1 Data Source
- **shibing624/huatuo_medical_qa_sharegpt:** 276,042 Chinese medical conversation pairs
- Covers: internal medicine, surgery, pediatrics, dermatology, etc.

#### 4.4.2 Data Processing Pipeline
- ChatML format conversion (system/user/assistant roles)
- System prompt: "你是一个专业的临床医生..."
- Token length filtering: max 1024 tokens (to prevent OOM on RTX 3060)
- Train/Eval split: 20,000 train + 1,000 evaluation samples

#### 4.4.3 Data Statistics
| Statistic | Value |
|-----------|-------|
| Total raw samples | 276,042 |
| After filtering (≤1024 tokens) | 272,985 (98.9%) |
| Training set | 20,000 |
| Evaluation set | 1,000 |
| Avg. tokens per sample | ~380 |

---

## 5. Experiments (1-1.5 pages)

### 5.1 Experimental Setup
- Hardware: NVIDIA RTX 3060 12GB, CUDA 12.1
- Software: PyTorch 2.5.1, Unsloth, Transformers 4.57.6, TRL
- Evaluation metrics and their rationale

### 5.2 Automatic Metrics

#### 5.2.1 Perplexity
- Evaluated on 500 held-out samples
- Measures model confidence in predicting medical text

#### 5.2.2 ROUGE-L & BLEU
- ROUGE-L: longest common subsequence (F1)
- BLEU: n-gram precision with smoothing
- Limitations: favor shorter, template-like responses

#### 5.2.3 BERTScore
- Semantic similarity using BERT embeddings
- More robust for dialogue evaluation
- Chinese BERT model for embedding

### 5.3 Human Evaluation

#### 5.3.1 Evaluation Protocol
- Blind A/B testing (evaluators unaware of model identity)
- 50 diverse medical questions sampled from test set
- Three criteria scored on 1-5 Likert scale:
  - **Accuracy:** Is the medical information correct?
  - **Completeness:** Does the answer cover key clinical points?
  - **Safety:** Does the answer include appropriate disclaimers?
- 2-3 evaluators with medical background

#### 5.3.2 Inter-rater Agreement
- Fleiss' kappa or Cohen's kappa

### 5.4 Qualitative Analysis
- 3-5 representative case studies with side-by-side comparisons
- Examples covering: diagnosis, medication, chronic disease management

### 5.5 Ablation Studies (optional)
- Effect of LoRA rank $r$ on performance
- Impact of training data size

---

## 6. Results (0.5-1 page)

### 6.1 Main Results

| Metric | Base | MedQwen (Ours) | Improvement |
|--------|------|----------------|-------------|
| Perplexity ↓ | 9.13 | **5.18** | **+43.2%** |
| ROUGE-L ↑ | 0.0940 | **0.1210** | **+28.8%** |
| BLEU ↑ | 0.0010 | **0.0012** | **+18.5%** |
| BERTScore F1 ↑ | 0.6845 | **0.7191** | **+5.0%** |

### 6.2 Human Evaluation Results

| Criterion | Base | MedQwen | Improvement |
|-----------|------|---------|-------------|
| Accuracy (1-5) | 2.81 | **3.24** | **+15.3%** |
| Completeness (1-5) | 1.70 | **1.72** | +1.2% |
| Safety (1-5) | 2.72 | **2.82** | +3.7% |
| **Overall** | 2.41 | **2.59** | **+7.5%** |
| **Preference** | 28% | **72%** | — |

### 6.3 Qualitative Examples
- Table with 3-4 comparison cases
- Highlight differences in diagnostic reasoning, treatment recommendations

### 6.4 Efficiency Analysis
- Training: 18.4M parameters, ~5 hours on RTX 3060
- Inference: real-time (< 4s per response)
- Memory: ~3GB during inference

---

## 7. Discussion (0.5 page)

### 7.1 Key Findings
- LoRA effectively transfers medical knowledge with minimal parameters
- BERTScore captures semantic improvement that ROUGE/BLEU miss
- Human evaluation confirms practical utility

### 7.2 Limitations
- 1.5B model may lack depth for complex clinical reasoning
- Evaluation on static test set, not real clinical deployment
- Safety concerns: model outputs should not replace professional diagnosis

### 7.3 Future Work
- Scale to larger models (7B, 14B)
- Incorporate multi-turn dialogue history
- Add retrieval-augmented generation (RAG) for evidence-based responses
- Clinical trial for real-world validation

---

## 8. Conclusion (0.25 page)
- Summary of contributions
- Impact for resource-constrained medical AI deployment
- Closing statement

---

## 9. References
- LoRA: Hu et al., ICLR 2022
- Qwen2: Yang et al., 2024
- HuatuoGPT: Zhang et al., 2023
- Unsloth: unsloth.ai
- BERTScore: Zhang et al., NeurIPS 2019
- ROUGE: Lin, ACL 2004
- Medical LLM evaluation standards

---

## Appendices (if page limits allow)
- A. Full hyperparameter configuration
- B. Additional qualitative examples
- C. Human evaluation instructions and rubric
