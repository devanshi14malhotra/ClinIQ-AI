# ClinIQ AI

Explainable, retrieval-based medical question answering chatbot built on NIH-sourced data. Retrieves answers from a verified dataset instead of generating them, eliminating hallucination risk. Not a diagnostic tool. For informational purposes only.

## Features

- 14,000+ medical QA pairs from NIH (cancer.gov, NIDDK, NINDS, GARD, MedlinePlus, and others)
- Dense semantic retrieval using sentence-transformers and FAISS, capturing meaning beyond exact keyword overlap
- Intent classification across 5 categories: symptom, treatment, cause, prevention, and general information
- LIME-based explainability on the intent classifier, providing word-level attribution weights for each prediction
- Confidence threshold of 0.45 so the system returns a no-match response instead of a low-confidence answer
- Answers displayed with punctuation and sentence structure preserved

## How it works

1. The query is classified by intent using a Logistic Regression model trained on TF-IDF features
2. LIME generates perturbed versions of the query, measures how the prediction changes across each, and produces per-word attribution weights explaining the intent decision
3. The query is encoded into a 384-dimensional dense vector using `all-MiniLM-L6-v2`
4. FAISS performs cosine similarity search over 14,301 pre-indexed question embeddings
5. If the best match score falls below 0.45, a no-match response is returned. Otherwise the answer is returned with a confidence label, LIME explanation, and alternative matches.

## Tech stack

| Component | Tool |
|---|---|
| Intent classification | Logistic Regression + TF-IDF (scikit-learn) |
| Explainability | LIME (LimeTextExplainer) |
| Semantic retrieval | sentence-transformers (all-MiniLM-L6-v2) + FAISS |
| Interface | Streamlit |
| Model persistence | joblib, numpy |

## v1 vs v2 retrieval comparison

| Query | v1 TF-IDF | v2 Dense |
|---|---|---|
| hypertension risks | What is Portal hypertension? (wrong) | Who is at risk for High Blood Pressure? (correct) |
| renal failure causes | What causes Heart Failure? (wrong) | What causes Kidney Failure? (correct) |
| how do I manage high blood sugar | What is High Blood Pressure? (wrong) | How to prevent Diabetes? (correct) |
| what drugs are used for asthma | What is Asthma? (partial) | What are the treatments for Asthma? (correct) |
| what is the best pizza topping | returned a low-confidence match (wrong) | correctly returned no-match |

v1 fails on vocabulary mismatch because TF-IDF relies on exact token overlap. "hypertension" and "high blood pressure" share no tokens so v1 scores them as unrelated. v2 encodes meaning so synonyms and paraphrases match correctly.

## Notebooks

`development.ipynb` covers the v1 build: data cleaning, TF-IDF vectorization, intent classifier training, and keyword-overlap explainability.

`expansion.ipynb` covers the v2 upgrades: improved intent labeling across 5 classes, classifier retraining on raw question text for LIME compatibility, dense embedding generation, FAISS index construction, LIME explainability, answer display fix, and a side-by-side comparison of v1 vs v2 retrieval across 10 test queries.

## Setup

```bash
pip install -r requirements.txt
```

Run `expansion.ipynb` top to bottom first to generate the v2 model files in `data/`.

```bash
streamlit run app.py
```

## Evaluation

Intent classifier (v2): 99% accuracy, weighted F1 0.99, evaluated on a held-out 20% test split across 5 classes.

Note: intent labels were generated using rule-based keyword matching. The high accuracy reflects how well the model learns those keyword patterns rather than deeper semantic intent understanding. This is documented for transparency.

## Known limitations

- Retrieval is restricted to topics covered by MedQuAD's 12 NIH source sites. Queries on conditions not present in the dataset may not retrieve relevant results even at high similarity scores.
- The system cannot synthesize information across multiple dataset entries or reason beyond retrieval.
- LIME attribution weights vary slightly across runs due to random perturbation sampling.

## Live demo

[cliniq-ai-dm.streamlit.app](https://cliniq-ai-dm.streamlit.app/)