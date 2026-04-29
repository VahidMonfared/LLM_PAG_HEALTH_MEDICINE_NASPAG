# PAG-Health-LLM: Domain-Specialized LLM for Pediatric & Adolescent Gynecology

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-yellow)](https://huggingface.co/VahidMonfared/PAG-Health-LLM-Mistral-7B)
[![Web App](https://img.shields.io/badge/Web%20App-Live-green)](https://gyno-pag-llm-health-med.lovable.app/)

---

## 🏥 Overview

**PAG-Health-LLM** is the first domain-specialized Large Language Model (LLM) for **Pediatric and Adolescent Gynecology (PAG)**, combining QLoRA fine-tuning with Retrieval-Augmented Generation (RAG) over the *NASPAG Essentials of Pediatric and Adolescent Gynecology (2025)* textbook.

Our 7-billion-parameter model **outperforms GPT-4o-mini, LLaMA-3.3-70B (10× larger), and Qwen-3-32B** across all evaluation metrics on 182 held-out clinical questions (all p < 0.001, Cohen's d = 0.89–1.70, large effects).

> ⚠️ **Research and educational tool only. Not a substitute for licensed medical judgment.**

---

## 🚀 Key Results

| Model | ROUGE-L | BERTScore |
|---|---|---|
| **PAG-Health-LLM (Ours, 7B)** | **0.413** | **0.909** |
| GPT-4o-mini | 0.232 | 0.870 |
| LLaMA-3.3-70B (70B) | 0.217 | 0.868 |
| Qwen-3-32B (32B) | 0.184 | 0.854 |

- ✅ +64% ROUGE-1 vs base model
- ✅ +374% BLEU vs base model
- ✅ +82% faithfulness (hallucination reduction)
- ✅ All p < 0.001, all Cohen's d > 0.8 (large effect)

---

## 🌐 Links

| Resource | URL |
|---|---|
| 🤗 Model (HuggingFace) | https://huggingface.co/VahidMonfared/PAG-Health-LLM-Mistral-7B |
| 🌐 Web App | https://gyno-pag-llm-health-med.lovable.app/ |
| 📄 Paper | *(under review)* |

---

## 🏗️ Architecture

**6-stage pipeline:**

1. PDF extraction from NASPAG textbook (30 chapters, 386 pages, ~195k tokens)
2. QA dataset generation with chapter-level no-leakage train/test split (70/15/15)
3. QLoRA fine-tuning of Mistral-7B-Instruct-v0.3 (4-bit NF4, rank=16, 3 epochs, ~30 min on T4 GPU)
4. RAG pipeline with BGE-small-en-v1.5 embeddings + FAISS index (250-word chunks, 50 overlap)
5. Cascading inference: RAG-first → model-fallback with out-of-corpus disclosure
6. Evaluation + deployment (HuggingFace Hub + Lovable web app)

---

## ⚙️ Quickstart

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base + fine-tuned adapter
base = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
model = PeftModel.from_pretrained(base, "VahidMonfared/PAG-Health-LLM-Mistral-7B")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
```

---

## 📊 Evaluation

| Metric | Description |
|---|---|
| ROUGE-1/2/L | N-gram overlap with gold answers |
| BLEU | Token-level precision |
| BERTScore F1 | Semantic similarity |
| Faithfulness | ROUGE-L vs source passage (hallucination proxy) |
| Cohen's d | Effect size for all pairwise comparisons |

**Test set:** 182 held-out questions from chapters never seen during training (no data leakage).

---

## 📁 Repository Contents
---

## 📋 Citation

If you use this work, please cite:

```bibtex
@article{monfared2025paghealth,
  title={A Specialist Outperforms Generalists: A Domain-Tuned 
         Retrieval-Augmented Large Language Model for Pediatric 
         and Adolescent Gynecology Clinical Decision Support},
  author={Monfared, Vahid},
  journal={under review},
  year={2025}
}
```

---

## 📜 License

Apache 2.0 — free for research, clinical education, and commercial use.

---

## ⚠️ Disclaimer

This system is a **research and clinical education tool** trained on the NASPAG Essentials textbook. It is not approved for direct patient care and does not replace the judgment of a licensed medical professional. Always consult a qualified physician for clinical decisions.
