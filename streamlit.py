import streamlit as st
import PyPDF2
import joblib
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
 
# Setup
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# BERT model path (if local)
bert = SentenceTransformer("bert_model")

# Load models
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
tfidf_model = joblib.load("models/logistic_regression_model.pkl")
bert_model = joblib.load("models/bert_logistic_model.pkl")

# Utils
def clean_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# UI
st.title("AI-Based Resume Screener System")
st.caption("Predicts best-fit job role from the PDF resume using ML models.")

model_choice = st.radio("Choose Model:", ["TF-IDF (LogReg)", "BERT (LogReg)"])

uploaded_file = st.file_uploader("Upload a PDF Resume", type=["pdf"])

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Resume Text", resume_text[:500], height=250)

    if model_choice == "TF-IDF (LogReg)":
        cleaned = clean_text(resume_text)
        features = tfidf_vectorizer.transform([cleaned])
        prediction = tfidf_model.predict(features)[0]
    else:
        embeddings = bert.encode([resume_text])
        prediction = bert_model.predict(embeddings)[0]

    st.success(f"Predicted Job Category: **{prediction}**")
