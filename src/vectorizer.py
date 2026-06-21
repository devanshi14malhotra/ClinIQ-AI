# ClinIQ AI - vectorizer.py
# Phase 2: TF-IDF vectorization + intent classifier training

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocess import preprocess_text


def train_intent_classifier(data_path='../data/medquad_preprocessed.csv'):
    df = pd.read_csv(data_path)

    X = df['question_processed']
    y = df['intent']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))

    joblib.dump(tfidf, '../data/tfidf_vectorizer.pkl')
    joblib.dump(model, '../data/intent_classifier.pkl')
    print("saved tfidf_vectorizer.pkl and intent_classifier.pkl")

    return tfidf, model


def load_intent_components():
    """Used by pipeline.py at inference time."""
    vectorizer = joblib.load('../data/tfidf_vectorizer.pkl')
    model = joblib.load('../data/intent_classifier.pkl')
    return vectorizer, model


def predict_intent(user_query, vectorizer, model):
    query_processed = preprocess_text(user_query)
    if not query_processed.strip():
        return 'unknown'
    vec = vectorizer.transform([query_processed])
    return model.predict(vec)[0]


if __name__ == '__main__':
    train_intent_classifier()