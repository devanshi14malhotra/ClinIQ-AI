# ClinIQ AI - retrieval.py
# Phase 3: TF-IDF retrieval engine using cosine similarity

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import preprocess_text


def build_retrieval_index(data_path='../data/medquad_preprocessed.csv'):
    df = pd.read_csv(data_path)

    retrieval_vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )
    question_vectors = retrieval_vectorizer.fit_transform(df['question_processed'])

    joblib.dump(retrieval_vectorizer, '../data/retrieval_vectorizer.pkl')
    joblib.dump(question_vectors, '../data/question_vectors.pkl')
    df.to_pickle('../data/cliniq_data.pkl')

    print("saved retrieval_vectorizer.pkl, question_vectors.pkl, cliniq_data.pkl")
    return retrieval_vectorizer, question_vectors, df


def load_retrieval_components():
    """Used by pipeline.py at inference time."""
    retrieval_vectorizer = joblib.load('../data/retrieval_vectorizer.pkl')
    question_vectors = joblib.load('../data/question_vectors.pkl')
    df = pd.read_pickle('../data/cliniq_data.pkl')
    return retrieval_vectorizer, question_vectors, df


def retrieve_answer(user_query, retrieval_vectorizer, question_vectors, df, top_k=3):
    """
    Takes a raw user query, returns top_k most similar QA pairs
    along with their similarity scores.
    """
    query_processed = preprocess_text(user_query)
    query_vector = retrieval_vectorizer.transform([query_processed])

    similarities = cosine_similarity(query_vector, question_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            'question': df.iloc[idx]['question'],
            'answer': df.iloc[idx]['answer_clean'],
            'focus_area': df.iloc[idx]['focus_area'],
            'intent': df.iloc[idx]['intent'],
            'similarity': similarities[idx],
            'index': idx
        })

    return results


if __name__ == '__main__':
    build_retrieval_index()