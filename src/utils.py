# src/utils.py
import re
import nltk
from nltk.corpus import stopwords

# Download once
nltk.download('stopwords', quiet=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)          # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)         # Remove special chars
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])
    return text
