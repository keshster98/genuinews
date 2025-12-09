# Fake News Detector  
A machine learning–based web application that classifies news articles as **Real** or **Fake** using writing style, linguistic patterns, and text structure. The project uses an NLP pipeline built with **NLTK** and a **Logistic Regression** classifier, deployed via an interactive **Gradio** web interface.

> **Important Note:**  
> This model does **not** perform fact-checking or verify information against external databases.  
> Predictions are based solely on **linguistic cues**, **writing patterns**, and **stylometric indicators**, not factual truth.

## Dataset  
🔗 **[Fake News Detection Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)**

This project uses publicly available fake and real Western news datasets from Kaggle. The dataset provides labeled real and fake news samples which were cleaned, vectorized, and used to train the Logistic Regression classifier.

## Features  
- Real-time news classification (Fake vs Real)  
- Probability-based confidence score  
- Custom NLP preprocessing pipeline with:
  - Lowercasing & noise removal  
  - URL, punctuation, and digit filtering  
  - Tokenization  
  - Stopword removal  
  - POS tagging via NLTK  
  - Lemmatization with POS awareness  
- Combines **title + article content** for richer input  
- Fully interactive Gradio interface

## How It Works  
### 1️⃣ Input  
User enters:
- News headline  
- News content  

### 2️⃣ NLP Processing  
Text is cleaned and normalized, then processed through:
- Tokenization  
- Stopword filtering  
- POS tagging  
- POS-aware lemmatization  

### 3️⃣ Model Prediction  
- The cleaned text is vectorized using a trained TF-IDF model  
- Logistic Regression outputs:
  - **Fake News** or **Real News**  
  - Confidence score (0–100%)

## Tech Stack  
- **Python**  
- **Gradio** (web UI)  
- **NLTK** (tokenization, stopwords, POS tagging, lemmatization)  
- **Scikit-learn** (TF-IDF & Logistic Regression)  
- **Joblib** (model loading)

## Live Demo 
**You can [try the app](https://huggingface.co/spaces/keshster98/genuinews)** directly in your web browser.

## Running Locally  

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/genuinews.git
pip install -r requirements.txt
python app.py
```
