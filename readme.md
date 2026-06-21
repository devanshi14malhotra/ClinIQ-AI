# ClinIQ AI

Explainable, retrieval-based medical question answering chatbot built on NIH-sourced data.

Rather than generating answers, it retrieves the closest matching answer from a verified dataset, removing the risk of hallucination common in generative chatbots, while exposing the reasoning behind each response through keyword-level explainability and confidence scoring. It is not a diagnostic tool and is not a substitute for professional medical advice.

## Features

- Retrieval-based question answering over 14,000+ medical QA pairs sourced from NIH websites (cancer.gov, NIDDK, NINDS, GARD, MedlinePlus, and others)
- Intent classification of user queries into symptom, treatment, or general information categories
- Explainability layer showing which keywords drove a match, with the matched terms highlighted directly in the returned answer
- Confidence scoring based on cosine similarity, with a no-match threshold so the system declines to answer rather than return an unreliable result
- Alternative match suggestions when more than one relevant answer exists in the dataset
- Streamlit web interface

## How it works

1. User submits a question through the Streamlit interface
2. The query is cleaned, tokenized, and lemmatized using the same pipeline applied to the dataset during preprocessing
3. An intent classifier (Logistic Regression trained on TF-IDF features) labels the query as a symptom query, treatment query, or general information query
4. The processed query is vectorized using TF-IDF and compared against all question vectors in the dataset using cosine similarity
5. If the best match falls below a similarity threshold, the system returns a no-match response instead of a low-confidence guess
6. If a confident match is found, matched keywords are identified and highlighted in the returned answer, alongside a confidence label and any alternative matches


## Tech stack

| Component | Tool |
|---|---|
| Preprocessing | pandas, NLTK, regex |
| Feature engineering | scikit-learn (TF-IDF) |
| Intent classification | scikit-learn (Logistic Regression, compared against Naive Bayes and Linear SVM) |
| Retrieval | Cosine similarity over TF-IDF vectors (scikit-learn) |
| Explainability | TF-IDF weighted keyword overlap, confidence banding |
| Interface | Streamlit |
| Model persistence | joblib |



## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run from the project root, not from inside `src/`.

## Evaluation

Intent classifier (Logistic Regression), evaluated on a held-out 20% test split:

- Accuracy: 99%
- Weighted F1: 0.99
- 5-fold cross-validation mean F1: 0.99 (std dev 0.002)

Note on the intent classifier: training labels were generated using rule-based keyword matching (presence of words like "symptom" or "treat" in the question). The classifier is trained on the same text used to create these labels, so the high accuracy reflects how well the model learns to detect those keyword patterns rather than a deeper semantic understanding of intent. This is documented for transparency.

Retrieval is evaluated qualitatively via similarity scores and a tuned no-match threshold (0.45), which was set by testing the system against both in-domain and out-of-domain queries.

## Known limitations

- Retrieval is restricted to terms present in the dataset's vocabulary. Queries containing terms absent from MedQuAD (for example, certain disease names not covered by the 12 source NIH sites) will not retrieve relevant results, even if the similarity score appears high. This is a known limitation of TF-IDF based retrieval and is mitigated, but not fully solved, by the no-match threshold.
- The system answers only what exists in the dataset. It cannot synthesize information across multiple entries or reason beyond retrieval.
- Intent classification reflects keyword presence rather than deep semantic intent (see Evaluation section above).

