import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import os

# Load features and labels
X = joblib.load("data/features.pkl")
y = joblib.load("data/labels.pkl")

# Defining the models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC()
}

# Create model folder if not exists
os.makedirs("models", exist_ok=True)

# Train + Evaluate all
for name, model in models.items():
    print(f"\nTraining {name}...\n")
    model.fit(X, y)
    y_pred = model.predict(X)

    print(f"Classification Report for {name}:\n")
    print(classification_report(y, y_pred))

    print(f"Confusion Matrix for {name}:\n")
    print(confusion_matrix(y, y_pred))

    # Save model
    model_path = f"models/{name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Saved {name} model to {model_path}")
