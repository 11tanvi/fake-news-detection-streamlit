Smart Fake News Detection
An end-to-end machine learning web application that detects fake vs. real news using NLP and domain-specific models. The system selects the best model based on news type (political statements vs. social-media/viral news) and provides an interactive Streamlit interface for real-time testing and explainability.

overview
Fake news spreads fast and influences public opinion; automated detection helps platforms and users flag likely misinformation for human review. This project demonstrates practical NLP, model selection, and deployment skills applied to a real-world problem.

Key features
Detects fake news from free-text input

Domain-aware models: separate models for political statements and social-media news.

Clear model performance metrics and comparison.

Interactive UI with prediction outputs and explanations.

End-to-end pipeline from preprocessing → training → evaluation → deployment.

Datasets
LIAR — political statements dataset (multi-class labels; used for political-domain model).

FakeNewsNet — social-media and viral news dataset (binary labels; used for social-media model).

Models & performance
Dataset	Model	Primary metric
LIAR (political)	XGBoost	Accuracy ≈ 91%
FakeNewsNet (social)	Logistic Regression	Accuracy ≈ 83%
Additional metrics (ROC-AUC, precision, recall, F1) and confusion matrices are available in the evaluation notebooks.

What I built
Data cleaning and text normalization (tokenization, stopword removal, lemmatization).

Feature engineering using TF‑IDF and domain-specific features.

Model selection and training pipelines (scikit-learn + XGBoost).

Model serialization (joblib) and inference wrapper.

dashboard for interactive predictions and basic explainability.

Deployment-ready repo with instructions and examples.

Demo
Live demo: [https://fake-news-detection-streamlit.onrender.com/]

Project structure
text
smart-fake-news-detection/
├─ app.py                  # Streamlit app
├─ notebooks/              # EDA & training notebooks
├─ models/                 # Trained model files (joblib)                 
├─ src/                    # preprocessing, training, inference modules
├─ assets/                 # screenshots, figures
├─ requirements.txt
└─ README.md

Example usage
Single prediction: paste an article or short statement → choose domain → Predict → get probability and label.

