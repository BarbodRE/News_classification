# 📰 News Classification with Transformers

This project demonstrates a **News Category Classification** task using the [AG News dataset](https://huggingface.co/datasets/ag_news).  
It applies **Hugging Face Transformers** with PyTorch to fine-tune a pre-trained model on four categories:

- 🌍 World  
- 🏅 Sports  
- 💼 Business  
- 🔬 Science/Technology  

---

## 📌 Project Overview

- **Data**: AG News dataset (training + test set)  
- **Model**: DistilBERT fine-tuned for text classification  
- **Frameworks**: PyTorch, Hugging Face Transformers, Datasets, Evaluate, Scikit-learn  
- **Metrics**: Accuracy, Precision, Recall, F1-score  
- **Extras**: Confusion Matrix + Misclassified Samples Analysis  

---

## 🚀 Results

| Metric     | Value |
|------------|-------|
| Accuracy   | ~85%  |
| Precision  | ~85%  |
| Recall     | ~85%  |
| F1-score   | ~84%  |

Confusion matrix and misclassified sample analysis are included in the notebook for error inspection.

---

## 📂 Repository Structure


News_Classification.ipynb # Main notebook with training, evaluation, and analysis

*(Optional future work: add `app.py` or `streamlit_app.py` for API / web demo.)*

---

## 🛠️ How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/news-classification.git
   cd news-classification


Install dependencies:
pip install torch transformers datasets evaluate scikit-learn


Open the notebook and run step by step:
jupyter notebook News_Classification.ipynb


🎯 Future Improvements

Hyperparameter tuning for better performance

Experimenting with larger models (BERT, RoBERTa, DeBERTa)

Deploying as an API (FastAPI/Flask) or interactive demo (Streamlit/HF Spaces)

✨ Demo (Notebook)

At the end of the notebook, there is a simple demo:
👉 Input a custom text and the model will predict its news category.

📜 License

This project is released under the MIT License.
