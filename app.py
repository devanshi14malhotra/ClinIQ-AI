# ClinIQ AI - app.py (v2)
# Uses dense retrieval (FAISS + sentence-transformers) and LIME explainability

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from lime.lime_text import LimeTextExplainer

st.set_page_config(page_title="ClinIQ AI", page_icon="🩺", layout="centered")


@st.cache_resource
def load_models():
    intent_pipeline = joblib.load('data/intent_pipeline_v2.pkl')
    df = pd.read_pickle('data/cliniq_data_v2.pkl')
    index = faiss.read_index('data/faiss_index_v2.bin')
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return intent_pipeline, df, index, embed_model


intent_pipeline, df, index, embed_model = load_models()

lime_explainer = LimeTextExplainer(class_names=list(intent_pipeline.classes_))


def explain_intent(user_query, num_features=6):
    exp = lime_explainer.explain_instance(
        user_query,
        intent_pipeline.predict_proba,
        num_features=num_features,
        labels=list(range(len(intent_pipeline.classes_)))
    )
    predicted_class = intent_pipeline.predict([user_query])[0]
    class_idx = list(intent_pipeline.classes_).index(predicted_class)
    attributions = exp.as_list(label=class_idx)
    attributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return attributions


def dense_retrieve(user_query, top_k=3, threshold=0.45):
    query_emb = embed_model.encode([user_query])
    query_norm = (query_emb / np.linalg.norm(query_emb)).astype('float32')
    scores, indices = index.search(query_norm, top_k)
    scores, indices = scores[0], indices[0]
    best_score = float(scores[0])

    if best_score < threshold:
        return {
            'status': 'no_match',
            'message': "I couldn't find a confident answer in my medical database. Try rephrasing, or consult a healthcare professional.",
            'best_attempt': {
                'question': df.iloc[indices[0]]['question'],
                'similarity': best_score
            }
        }

    results = []
    for score, idx in zip(scores, indices):
        results.append({
            'question': df.iloc[idx]['question'],
            'answer': df.iloc[idx]['answer_display'],
            'focus_area': df.iloc[idx]['focus_area'],
            'similarity': float(score),
        })
    return {'status': 'match_found', 'results': results}


def confidence_label(score, threshold=0.45):
    if score < threshold:
        return 'No reliable match', '🔴'
    elif score < 0.6:
        return 'Medium', '🟡'
    else:
        return 'High', '🟢'


# ── UI ───────────────────────────────────────────────────────────

st.title("🩺 ClinIQ AI")
st.caption("Explainable medical Q&A · retrieval-based · NIH-sourced data")
st.warning("⚠️ Not a substitute for professional medical advice. For informational purposes only.")

user_query = st.text_input(
    "Ask a medical question:",
    placeholder="e.g. what are the symptoms of diabetes"
)

if user_query:
    if not user_query.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Searching..."):
            predicted_intent = intent_pipeline.predict([user_query])[0]
            intent_explanation = explain_intent(user_query)
            retrieval_result = dense_retrieve(user_query)

        st.markdown(f"**Predicted intent:** `{predicted_intent}`")

        if retrieval_result['status'] == 'no_match':
            score = retrieval_result['best_attempt']['similarity']
            label, emoji = confidence_label(score)
            st.markdown(f"**Confidence:** {emoji} {label} ({score:.3f})")
            st.error(retrieval_result['message'])
            st.caption(f"Closest match found: *{retrieval_result['best_attempt']['question']}*")

        else:
            best = retrieval_result['results'][0]
            score = best['similarity']
            label, emoji = confidence_label(score)
            st.markdown(f"**Confidence:** {emoji} {label} ({score:.3f})")

            st.subheader("Answer")
            st.write(best['answer'])

            with st.expander("Why this answer? (Explainability)"):
                st.write(f"**Matched question in database:** {best['question']}")
                st.write(f"**Focus area:** {best['focus_area']}")
                st.write(f"**Similarity score:** {score:.4f}")

                st.write("**Intent explanation (LIME):**")
                for word, weight in intent_explanation:
                    bar = "🟩" if weight > 0 else "🟥"
                    st.write(f"{bar} `{word}` — {weight:.4f}")

                if len(retrieval_result['results']) > 1:
                    st.write("**Other possible matches:**")
                    for r in retrieval_result['results'][1:]:
                        st.write(f"- {r['question']} (sim: {r['similarity']:.3f})")