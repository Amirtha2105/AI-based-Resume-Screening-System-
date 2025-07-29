import nltk
import spacy
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)


nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text) # Tokenization
    tokens = [word for word in tokens if word not in stop_words] # Stop Words Removal
    tokens = [stemmer.stem(word) for word in tokens] # Stemming
    return " ".join(tokens)

# Load data
df = pd.read_csv("D://PUP.pup//INTERNSHIP 3//AI_based_resume_screening_system//data//Resume.csv")  

# Preprocessing
df["Cleaned"] = df["Resume_str"].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["Cleaned"])
y = df["Category"]

# Before saving
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Save vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(X, "data/features.pkl")
joblib.dump(y, "data/labels.pkl")
