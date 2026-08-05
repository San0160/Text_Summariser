# 🚀 End-to-End Text Summariser (Production Ready)

> **An end-to-end NLP application for dialogue summarization using
> Google's FLAN-T5 model, built with a modular MLOps pipeline and
> deployed on Railway.**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-green)
![Docker](https://img.shields.io/badge/Docker-blue)
![Railway](https://img.shields.io/badge/Deployed-Railway-purple)

------------------------------------------------------------------------

## 🌐 Live Demo

**Railway URL:**

`https://textsummariser-production.up.railway.app`

**GitHub Repository**

`https://github.com/San0160/Text_Summariser`

------------------------------------------------------------------------

# ✨ Highlights

-   🤖 FLAN-T5 based text summarization
-   ⚙️ Modular MLOps pipeline
-   📊 ROUGE evaluation
-   🐳 Dockerized deployment
-   🚀 Hosted on Railway
-   🔁 CI/CD ready

------------------------------------------------------------------------

# 📌 Overview

This project performs abstractive dialogue summarization using a
T5-family transformer and follows a production-style MLOps workflow:

1.  Data Ingestion
2.  Data Validation
3.  Data Transformation
4.  Model Training
5.  Model Evaluation
6.  FastAPI Deployment
7.  Docker
8.  Railway

------------------------------------------------------------------------

# 🧠 Example

## Input

``` text
Amanda: I baked cookies.
Jerry: Sure!
Amanda: I'll bring you tomorrow.
```

## Output

``` text
Amanda baked cookies and will bring Jerry some tomorrow.
```

------------------------------------------------------------------------

# 📊 Results

  Metric             Value
  --------------- --------
  ROUGE-1           0.4119
  ROUGE-2           0.1860
  ROUGE-L           0.3345
  Training Loss     0.4019

------------------------------------------------------------------------

# 🏗 Architecture

``` text
Dataset
  ↓
Validation
  ↓
Transformation
  ↓
Training
  ↓
Evaluation
  ↓
FastAPI
  ↓
Docker
  ↓
Railway
```

------------------------------------------------------------------------

# 🛠 Tech Stack

-   Python
-   PyTorch
-   Hugging Face Transformers
-   FastAPI
-   Docker
-   Railway

------------------------------------------------------------------------

# 🚀 Run

``` bash
git clone https://github.com/San0160/Text_Summariser.git
cd Text_Summariser
pip install -r requirement.txt
python app.py
```

------------------------------------------------------------------------

# 🐳 Docker

``` bash
docker build -t text-summariser .
docker run -p 8080:8080 text-summariser
```

------------------------------------------------------------------------

# 🌍 Deployment

The application is containerized with Docker and deployed on Railway for
public access.

------------------------------------------------------------------------

# 📸 Screenshots

Add screenshots of: - Home page - Summary page - Mobile view

------------------------------------------------------------------------

# 🔮 Future Work

-   Hugging Face Hub integration
-   Quantization
-   ONNX optimization
-   Long-document summarization

------------------------------------------------------------------------

# 👨‍💻 Author

**Sandeep Kumar**

GitHub: https://github.com/San0160
