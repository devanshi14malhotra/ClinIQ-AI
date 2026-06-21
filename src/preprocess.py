# ClinIQ AI - preprocess.py
# Text cleaning, tokenization, lemmatization, intent labeling

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words -= {'no', 'not', 'nor', 'never', 'without', 'more', 'less', 'above', 'below'}


def assign_intent(question):
    """Rule-based intent labeling, used to bootstrap training labels."""
    q = question.lower()
    if any(w in q for w in ['symptom', 'sign', 'feel', 'pain', 'ache', 'fever', 'cough']):
        return 'symptom_query'
    elif any(w in q for w in ['treat', 'cure', 'medication', 'drug', 'therapy', 'medicine', 'dose']):
        return 'treatment_query'
    else:
        return 'general_info'


def clean_text(text):
    """Lowercase, strip URLs/HTML/brackets/special chars, normalize whitespace."""
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-z0-9\s\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text):
    """Full pipeline: clean -> tokenize -> remove stopwords -> lemmatize."""
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 2
    ]
    return ' '.join(tokens)