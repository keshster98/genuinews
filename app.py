import gradio as gr
import joblib
import re
import string
import numpy as np

# NLTK imports for cleaner
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Download required NLTK resources (only needs to run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# === Load model and vectorizer ===
model = joblib.load("models/lr_balanced_1.pkl")
vectorizer = joblib.load("models/lr_vectorizer_balanced_1.pkl")

# === Helper to convert POS tags ===
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# === Content Cleaning Function ===
def content_cleaner(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\w*\d\w*', '', text)
    tokens = word_tokenize(text)
    tokens = [
        word for word in tokens
        if word not in stopwords.words('english')
        and word not in string.punctuation
        and word.isalpha()
    ]
    pos_tags = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags
    ]
    return ' '.join(lemmatized_tokens)

# === Prediction Function ===
def classify_news(title, content):
    combined_text = f"{title} {content}"
    cleaned_text = content_cleaner(combined_text)

    vectorized_input = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_input)[0]
    proba = model.predict_proba(vectorized_input)[0]

    label = "Real News" if prediction == 1 else "Fake News"
    confidence = f"{np.max(proba)*100:.2f}%"
    return label, confidence

# === Gradio UI ===
title_input = gr.Textbox(label="News Title", placeholder="Enter the news headline...")
content_input = gr.Textbox(label="News Content", placeholder="Paste the full article text here...", lines=10)

label_output = gr.Textbox(label="Prediction")
confidence_output = gr.Textbox(label="Confidence Score")

interface = gr.Interface(
    fn=classify_news,
    inputs=[title_input, content_input],
    outputs=[label_output, confidence_output],
    title="Fake News Detector",
    description="Enter a news title and content to check if it is likely to be real or fake."
)

if __name__ == "__main__":
    interface.launch()
