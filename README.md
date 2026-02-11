Smart Fake News Detection

A machine learning–based web application to detect fake vs real news using textual analysis.
The project uses two real-world datasets and selects the most suitable model based on the news type.

Built with Python, NLP, and Streamlit.

Features

Detects fake news from text input

Supports political statements and social media news

Uses separate ML models for different news domains

Interactive Streamlit web interface

Datasets Used

LIAR Dataset – Political statements (multi-class classification)

FakeNewsNet Dataset – Social media & viral news (binary classification)

Models & Performance
Dataset 	         Model              	Accuracy
LIAR	            XGBoost	              ~91%
FakeNewsNet   	Logistic Regression	    ~83%
