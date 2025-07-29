# AI-based-Resume-Screening-System-

## AI-based Resume Screening System

An NLP-driven project to classify resumes into appropriate job categories using machine learning. Designed to streamline the hiring pipeline by helping HR teams filter relevant candidates quickly.

## Problem Statement

Manually reviewing thousands of resumes is time-consuming and inefficient. This project builds a model that reads the resume content, understands it using NLP techniques, and suggests the most suitable job role category.

## Tech Stack

- **Language**: Python  
- **Libraries**: NLTK, SpaCy, Scikit-learn, BERT, FastText, Streamlit  
- **Text Features**: TF-IDF, Word Embeddings  
- **IDE**: Jupyter Notebook, VS Code

## Project Timeline (4 Weeks)

### Week 1 - Data Collection & Preprocessing
- Gathered text-based resumes from datasets or simulated entries
- Cleaned data: tokenization, stopword removal, stemming/lemmatization
- Converted text to numeric features using **TF-IDF Vectorization**

### Week 2 - ML Model Building
- Trained classification models:
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Logistic Regression
- Evaluated with metrics: **Precision**, **Recall**, **Confusion Matrix**

### Week 3 - Advanced Embeddings
- Integrated **FastText** and optionally **BERT** for contextual embeddings
- Fine-tuned embeddings to improve classification accuracy

### Week 4 - Interface & Deployment
- Built a **Streamlit-based resume uploader**
- Model outputs top-matching job roles instantly
- Created a presentation/demo deck for HR demonstration

##  Features

- Text classification using TF-IDF or embeddings
- Upload resume in `.txt` or `.pdf` (with extraction)
- Predict job role like “Software Engineer”, “Data Analyst”, etc.
- Easy-to-use Streamlit dashboard

## Outputs

- A trained model for resume classification
- Streamlit web app for HR use-case
- Sample resumes tested and visualized for prediction performance

## Folder Structure

AI_based_resume_screening_system/
│
├── data/
│   ├── Resume.csv                   # Original dataset
│   ├── features.pkl                 # TF-IDF features
│   ├── labels.pkl                   # Labels for TF-IDF
│   ├── bert_features.pkl            # BERT-based features
│   ├── bert_labels.pkl              # Labels for BERT
│
├── models/
│   ├── tfidf_vectorizer.pkl         # TF-IDF vectorizer object
│   ├── naive_bayes_model.pkl        # Trained Naive Bayes model
│   ├── logistic_regression_model.pkl# Trained Logistic Regression (TF-IDF)
│   ├── linear_svm_model.pkl         # Trained Linear SVM
│   ├── bert_logistic_model.pkl      # Trained BERT + Logistic Regression
│
├── bert_model/                      # BERT is downloaded locally
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   ├── sentence_bert_config.json
│
├── streamlit_app/
│   └── streamlit.py                       # Streamlit frontend for resume screening
│
├── notebooks/  
    └── vectorize.py                     # Week 1: Preprocessing + TF-IDF vectorizer
├── model.py                         # Week 2: Training + Evaluation (TF-IDF models)
├── bert_vectorize.py                # Week 3: BERT embedding + model training
├── AI_Resume_Screener_DemoDeck.pptx # Week 4: Presentation file for demo
│
├── requirements.txt                 # (Optional) Python packages list
└── README.md                        # (Optional) Project overview for GitHub

