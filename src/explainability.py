# ClinIQ AI - explainability.py
# Phase 4: keyword matching, answer highlighting, confidence labeling

import re
from preprocess import preprocess_text


def get_matched_keywords(user_query, matched_question_idx, df, retrieval_vectorizer, top_n=5):
    """
    Finds which words from the user query overlap with the matched question,
    ranked by their TF-IDF importance.
    """
    query_processed = preprocess_text(user_query)
    query_tokens = set(query_processed.split())

    matched_question_processed = df.iloc[matched_question_idx]['question_processed']
    matched_tokens = set(matched_question_processed.split())

    overlap = query_tokens & matched_tokens
    if not overlap:
        return []

    feature_names = retrieval_vectorizer.get_feature_names_out()
    vocab = {word: idx for idx, word in enumerate(feature_names)}

    query_vector = retrieval_vectorizer.transform([query_processed])
    query_array = query_vector.toarray()[0]

    scored_overlap = []
    for word in overlap:
        if word in vocab:
            score = query_array[vocab[word]]
            scored_overlap.append((word, score))

    scored_overlap.sort(key=lambda x: x[1], reverse=True)
    return scored_overlap[:top_n]


def highlight_answer(answer_text, keywords):
    """
    Wraps matched keywords in the answer with ** markers
    so they render bold in Streamlit markdown.
    """
    highlighted = answer_text
    keyword_list = [kw for kw, score in keywords]

    for kw in keyword_list:
        pattern = r'\b(' + re.escape(kw) + r')\b'
        highlighted = re.sub(pattern, r'**\1**', highlighted, flags=re.IGNORECASE)

    return highlighted


def confidence_label(similarity_score, no_match_threshold=0.45):
    """Converts raw cosine similarity into a human-readable confidence band."""
    if similarity_score < no_match_threshold:
        return "No reliable match", "🔴"
    elif similarity_score < 0.6:
        return "Medium", "🟡"
    else:
        return "High", "🟢"


def get_explainable_answer(user_query, retrieval_vectorizer, question_vectors, df,
                            retrieve_answer_fn, threshold=0.45):
    """
    Full pipeline: query -> retrieval -> explainability -> structured response.
    retrieve_answer_fn is passed in from retrieval.py to avoid circular imports.
    """
    if not user_query or not user_query.strip():
        return {'status': 'invalid_input', 'message': "Please enter a question."}

    results = retrieve_answer_fn(user_query, retrieval_vectorizer, question_vectors, df, top_k=3)
    best = results[0]
    score = best['similarity']
    label, emoji = confidence_label(score, threshold)

    if score < threshold:
        return {
            'status': 'no_match',
            'confidence_label': label,
            'confidence_emoji': emoji,
            'similarity': score,
            'message': "I couldn't find a confident answer to that in my medical database. Try rephrasing, or consult a healthcare professional.",
            'closest_question': best['question'],
        }

    keywords = get_matched_keywords(user_query, best['index'], df, retrieval_vectorizer)
    highlighted_answer = highlight_answer(best['answer'], keywords)

    return {
        'status': 'match_found',
        'confidence_label': label,
        'confidence_emoji': emoji,
        'similarity': round(float(score), 4),
        'matched_question': best['question'],
        'focus_area': best['focus_area'],
        'intent': best['intent'],
        'answer': best['answer'],
        'answer_highlighted': highlighted_answer,
        'matched_keywords': [kw for kw, s in keywords],
        'alternative_matches': [
            {'question': r['question'], 'similarity': round(float(r['similarity']), 4)}
            for r in results[1:]
        ]
    }