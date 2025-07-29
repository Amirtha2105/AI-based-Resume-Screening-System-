# bert - vectorize and train

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import joblib
import os

# Load CSV
df = pd.read_csv("data/Resume.csv")
texts = df["Resume_str"].tolist()
labels = df["Category"].tolist()

# Load pre-trained BERT model
bert = SentenceTransformer("bert_model")

# BERT Embeddings
print("Encoding resumes using BERT...")
X = bert.encode(texts, show_progress_bar=True)

# Save embeddings and labels
os.makedirs("data", exist_ok=True)
joblib.dump(X, "data/bert_features.pkl")
joblib.dump(labels, "data/bert_labels.pkl")

# Train classifier
print("\nTraining classifier on BERT features...\n")
clf = LogisticRegression(max_iter=1000)
clf.fit(X, labels)

# Predict + Eval
y_pred = clf.predict(X)
print("Classification Report:\n")
print(classification_report(labels, y_pred))
print("Confusion Matrix:\n")
print(confusion_matrix(labels, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/bert_logistic_model.pkl")
