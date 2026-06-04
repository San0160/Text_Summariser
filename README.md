# 🗒️ End-to-End Text Summariser

An end-to-end NLP pipeline that fine-tunes a **T5 transformer** on the SAMSum dialogue dataset to automatically generate concise summaries of conversations. Built with a modular MLOps pipeline covering data ingestion through model deployment.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2+cu121-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.78.0-green)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

---

## 📌 Project Overview

This project fine-tunes Google's **T5-small** (60M parameters) on the [SAMSum dataset](https://huggingface.co/datasets/samsum) — a collection of 16,000 messenger-style dialogues paired with human-written summaries. The goal is to automatically compress multi-turn conversations into a single concise summary.

**Example:**

```
Input Dialogue:
  Amanda: I baked cookies. Do you want some?
  Jerry: Sure!
  Amanda: I'll bring you tomorrow :-)

Generated Summary:
  Amanda baked cookies and will bring Jerry some tomorrow.
```

---

## 📊 Results

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.4119 |
| ROUGE-2 | 0.1860 |
| ROUGE-L | 0.3345 |
| ROUGE-Lsum | 0.3343 |
| Training Loss | 0.4019 |
| Model Size | 60M parameters |
| Training Time | ~2.5 hours (NVIDIA GPU, fp16) |

> These are competitive scores for a 60M parameter model. Larger models like BART-large typically score ~0.52 on ROUGE-1.

---

## 🏗️ Project Architecture

```
Text_Summariser/
├── .github/workflows/        # CI/CD pipeline
├── config/
│   └── config.yaml           # Infrastructure config (paths, model checkpoints)
├── src/Text_summariser/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── pipeline/
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_data_validation.py
│   │   ├── stage_03_data_transformation.py
│   │   ├── stage_04_model_trainer.py
│   │   └── stage_05_model_evaluation.py
│   ├── config/
│   │   └── configuration.py
│   ├── entity/               # Dataclasses for configs
│   ├── constants/            # File paths
│   └── logging/              # Custom logger
├── reseach/                  # Jupyter notebooks for experimentation
├── app.py                    # FastAPI deployment
├── main.py                   # Full pipeline runner
├── params.yaml               # Training hyperparameters
├── requirement.txt
├── setup.py
├── dockerfile
└── template.py
```

---

## 🔄 ML Pipeline Stages

Each stage skips automatically if artifacts already exist — no redundant recomputation.

```
Stage 1: Data Ingestion       → Downloads/loads SAMSum CSV files
         ↓
Stage 2: Data Validation      → Checks schema, null values, column names
         ↓
Stage 3: Data Transformation  → Tokenises dialogues with T5 tokenizer
         ↓
Stage 4: Model Training       → Fine-tunes T5-small (skips if model exists)
         ↓
Stage 5: Model Evaluation     → Computes ROUGE scores on test set (skips if metrics.csv exists)
         ↓
Deployment: FastAPI           → Serves predictions via REST API
```

---

## 🧠 Advanced Techniques Used

| Technique | Description |
|-----------|-------------|
| **Transfer Learning** | Fine-tuned Google's T5-small pretrained on 750GB of text |
| **Mixed Precision (fp16)** | Half-precision GPU training — 2x faster with same accuracy |
| **Learning Rate Warmup** | Gradual LR increase then decay for stable training |
| **Gradient Accumulation** | Effective batch size of 128 without extra GPU memory |
| **Beam Search** | Explores 8 candidate summaries, picks the best one |
| **ROUGE Evaluation** | Industry standard metric for summarisation quality |
| **Modular MLOps Pipeline** | Separate stages with artifact caching and skip logic |

---

## ⚙️ Pipeline Update Workflow

When adding a new stage or modifying an existing one:

```
1. Update config.yaml          # paths and model checkpoints
2. Update params.yaml          # training hyperparameters
3. Update entity/              # dataclass for new config
4. Update configuration.py     # config manager method
5. Update components/          # business logic
6. Update pipeline/            # orchestration with skip logic
7. Update main.py              # add new stage
8. Update app.py               # if deployment changes
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10
- NVIDIA GPU with CUDA 12.1 (for training)
- Anaconda or virtualenv

### Installation

```bash
# Clone the repository
git clone https://github.com/San0160/Text_Summariser.git
cd Text_Summariser

# Create conda environment
conda create -n textS python=3.10
conda activate textS

# Install dependencies
pip install -r requirement.txt
```

### Running the Pipeline

```bash
# Run full pipeline (skips completed stages automatically)
python main.py
```

### Running the API

```bash
python app.py
```

Then visit `http://localhost:8080` to use the summariser.

### API Usage

```python
import requests

response = requests.post("http://localhost:8080/predict", 
    json={"text": "Amanda: I baked cookies. Jerry: Sure! Amanda: I'll bring you tomorrow"})

print(response.json())
# {"summary": "Amanda baked cookies and will bring Jerry some tomorrow."}
```

---

## 🐳 Docker

```bash
# Build image
docker build -t text-summariser .

# Run container
docker run -p 8080:8080 text-summariser
```

---

## 📚 What I Learned

**Deep Learning & NLP**
- How transformer encoder-decoder architecture works (T5, attention mechanism)
- Subword tokenisation — why models split rare words into pieces
- Beam search vs greedy decoding for text generation
- ROUGE metrics — ROUGE-1, ROUGE-2, ROUGE-L and when to use each

**Training Techniques**
- Transfer learning — leveraging pretrained weights for faster, better training
- Mixed precision training (fp16) for GPU efficiency
- Learning rate scheduling with warmup steps
- Gradient accumulation for effective large batch training
- Why grad_norm matters for training stability

**MLOps & Engineering**
- Separating infrastructure config (config.yaml) from hyperparameters (params.yaml)
- Building modular pipelines with artifact caching
- Why notebooks are for experimentation, pipelines are for production
- Model serialisation — safetensors vs pytorch_model.bin
- CUDA setup and GPU memory management

**Tools & Ecosystem**
- HuggingFace Transformers, Datasets, Evaluate libraries
- FastAPI for model serving
- Docker for containerisation
- Apache Arrow format for efficient dataset storage

---

## 🔮 Future Improvements

- Push model weights to HuggingFace Hub for easy sharing (Better update, pending)
- Auto-download artifacts from HuggingFace if not found locally (Pending)
- Upgrade to `t5-base` or `BART-large` for better ROUGE scores (Pending)
- Add CI/CD with GitHub Actions for automated retraining 
- Support for longer documents (articles, meeting transcripts)
- Add a web UI with Gradio or Streamlit

---

## 📄 License

This project is licensed under the Apache 2.0 License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [SAMSum Dataset](https://huggingface.co/datasets/samsum) — Samsung Research
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [T5 Paper](https://arxiv.org/abs/1910.10683) — Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
