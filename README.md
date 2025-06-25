# 📄 Multimodal Resume Shortlisting System

This project implements a deep learning pipeline for automating the shortlisting of resumes using **both structured tabular features** (like degree, experience) and **unstructured text** (the resume content). The goal is to build a proof-of-concept model capable of classifying whether a candidate should be shortlisted.

---

## 🚀 Overview

- Combines **tabular** and **textual** data using a **fusion model**
- Tabular branch: MLP classifier trained on structured features
- Textual branch: LSTM classifier trained on cleaned and tokenized resumes
- Labels are **synthetically generated** using rule-based heuristics
- Trained and evaluated on a cleaned dataset of 962 resumes

---

## 🧠 Model Architecture
Resume Text → LSTM → Feature Vector (64D)
Tabular Data → MLP → Feature Vector (64D)
↓
[Concat] → Fusion MLP → Output (shortlisted or not)


---

## 📊 Performance

| Model                | Accuracy | Weighted F1 | AUC   |
|---------------------|----------|-------------|-------|
| Fusion (LSTM + MLP) | 1.00     | 1.00        | 0.96  |
| Logistic Regression | 0.98     | 0.98        | 0.97  |
| SVC                 | 0.98     | 0.98        | 0.97  |
| Random Forest       | 0.95     | 0.95        | 0.94  |
| XGBoost             | 0.95     | 0.95        | 0.94  |

⚠️ Note: Labels were synthesized using domain rules; high performance reflects internal consistency rather than generalization to real recruiter judgments.

---

## 🧹 Preprocessing

- Tabular:
  - Extracted degree, field of study, job field, and experience
  - One-hot + standard scaling + label encoding
- Text:
  - Lowercased, lemmatized, tokenized
  - Truncated to 200 tokens
  - Indexed with custom vocabulary (min freq = 2)
  - Padded sequences

---

## 🧪 Features

- Custom label synthesis via resume keyword rules
- Flexible training pipeline (PyTorch)
- Evaluation with precision, recall, F1-score, confusion matrix, and ROC-AUC
- Modular design for extension or real-world adaptation

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- NLTK
- pandas, numpy, matplotlib

---

## 🏁 How to Run

This repo is mostly just a demo run, the full demo will be avaialable on Hugging Face soon ;)

---

## 📌 Future Work

- Deploy using [🤗 Hugging Face Spaces](https://huggingface.co/spaces)
- Replace synthetic labels with real annotations
- Add interpretability (attention, SHAP)
- Explore transformer-based resume encoders

---

## 📝 License

This project is open for educational use. Attribution appreciated if reused.
