# ClinIQ AI - Streamlit App
# Loads models directly, no relative path dependency on src/ folder location

import streamlit as st
import joblib
import pandas as pd
import sys
import os

# add src/ to path so we can import the logic functions (not the loaders)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import preprocess_text
from retrieval import retrieve_answer
from explainability import get_matched_keywords, highlight_answer, confidence_label
from vectorizer import predict_intent

st.set_page_config(page_title="ClinIQ AI", page_icon="🩺", layout="centered")


# ── Load models once, cache across reruns ──────────────────────
@st.cache_resource
def load_models():
    intent_vectorizer = joblib.load('data/tfidf_vectorizer.pkl')
    intent_model = joblib.load('data/intent_classifier.pkl')
    retrieval_vectorizer = joblib.load('data/retrieval_vectorizer.pkl')
    question_vectors = joblib.load('data/question_vectors.pkl')
    df = pd.read_pickle('data/cliniq_data.pkl')
    return intent_vectorizer, intent_model, retrieval_vectorizer, question_vectors, df


intent_vectorizer, intent_model, retrieval_vectorizer, question_vectors, df = load_models()


# ── UI ───────────────────────────────────────────────────────────
st.title("🩺 ClinIQ AI")
st.caption("Explainable medical Q&A, retrieval-based, NIH-sourced data")

st.warning("⚠️ Not a substitute for professional medical advice. For informational purposes only.")

user_query = st.text_input("Ask a medical question:", placeholder="e.g. what are the symptoms of diabetes")

if user_query:
    if not user_query.strip():
        st.error("Please enter a question.")
    else:
        predicted_intent = predict_intent(user_query, intent_vectorizer, intent_model)

        results = retrieve_answer(user_query, retrieval_vectorizer, question_vectors, df, top_k=3)
        best = results[0]
        score = best['similarity']
        label, emoji = confidence_label(score, no_match_threshold=0.45)

        st.markdown(f"**Predicted intent:** `{predicted_intent}`")
        st.markdown(f"**Confidence:** {emoji} {label} ({score:.3f})")

        if score < 0.45:
            st.error("I couldn't find a confident answer to that in my medical database. Try rephrasing, or consult a healthcare professional.")
            st.caption(f"Closest match found: *{best['question']}*")
        else:
            keywords = get_matched_keywords(user_query, best['index'], df, retrieval_vectorizer)
            highlighted = highlight_answer(best['answer'], keywords)

            st.subheader("Answer")
            st.markdown(highlighted)

            with st.expander("Why this answer? (Explainability)"):
                st.write(f"**Matched question in database:** {best['question']}")
                st.write(f"**Focus area:** {best['focus_area']}")
                st.write(f"**Matched keywords:** {', '.join([kw for kw, s in keywords])}")
                st.write(f"**Similarity score:** {score:.4f}")

                if len(results) > 1:
                    st.write("**Other possible matches:**")
                    for r in results[1:]:
                        st.write(f"- {r['question']} (sim: {r['similarity']:.3f})")