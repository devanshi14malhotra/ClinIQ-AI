# ClinIQ AI Development Notebook Analysis

Detailed breakdown and summary of each cell in `development.ipynb` notebook:

## 1. Preprocessing Phase
*   **[0] Markdown:** Header for preprocessing.
*   **[1] Code:** Installs required Python libraries using `subprocess.run()`. *(Correct: While `!pip install` is more common in notebooks, this subprocess approach works perfectly.)*
*   **[2] Code:** Imports essential libraries (pandas, numpy, regex, nltk, matplotlib) and downloads specific NLTK corpora (stopwords, punkt, wordnet). *(Correct)*
*   **[3] Code:** Loads the `medquad-dataset.csv` into a pandas DataFrame and prints its shape and columns. *(Correct)*
*   **[4] Code:** Performs Exploratory Data Analysis (EDA) showing statistics, null counts, data types, and source distributions. *(Correct)*
*   **[5] Code:** Calculates answer lengths and visualizes their distribution using a histogram to identify a cutoff point for overly short, uninformative answers. *(Correct)*
*   **[6] Code:** Cleans the data by dropping rows with missing answers, filling missing focus areas, and dropping temporary columns. *(Correct)*
*   **[7] Code:** Uses a rule-based keyword matching approach to label questions into three intents: `symptom_query`, `treatment_query`, and `general_info`. Plots the distribution. *(Correct: Great starting point for creating training labels.)*
*   **[8] Code:** Defines the `clean_text` function to strip URLs, HTML tags, brackets, and special characters, and lowercase the text. *(Correct)*
*   **[9] Code:** Defines the `preprocess_text` function, which tokenizes, removes stopwords (carefully keeping medically relevant words like 'no' and 'not'), and lemmatizes words to their roots. *(Correct: Excluding negation words from stopwords is a critical best practice in medical NLP.)*
*   **[10] Code:** Applies preprocessing to the entire dataset, removes duplicates, filters out short answers (length < 50), and saves the cleaned data to `medquad_preprocessed.csv`. *(Correct)*

## 2. Intent Classification Phase
*   **[11] Markdown:** TF-IDF Vectorisation header.
*   **[12] Code:** Reloads the preprocessed dataset. *(Correct: Good practice to decouple phases.)*
*   **[13] Code:** Assigns features (`X`) and targets (`y`) for the supervised learning model. *(Correct)*
*   **[14] Code:** Splits the data into 80% training and 20% testing sets. Uses `stratify=y` to preserve the class balance. *(Correct)*
*   **[15] Code:** Initializes a `TfidfVectorizer` (max 5000 features, using unigrams and bigrams), fits/transforms the training data, and transforms the test data. Fills NaNs before vectorizing. *(Correct: Avoids data leakage by not fitting on test data.)*
*   **[16] Code:** Trains and compares three baseline models: Logistic Regression, Naive Bayes, and Linear SVM. *(Correct)*
*   **[17] Code:** Generates a detailed classification report and confusion matrix plot for the best model (Logistic Regression). *(Correct)*
*   **[18] Code:** Performs 5-fold cross-validation on the training set using weighted F1-score to validate model stability. *(Correct)*
*   **[19] Code:** Saves both the TF-IDF vectorizer and the trained Logistic Regression model using `joblib`. *(Correct)*

## 3. Retrieval Engine Phase
*   **[20] Markdown:** Retrieval engine header.
*   **[21] Code:** Cleans up empty/NaN processed questions and creates a separate `TfidfVectorizer` (broader vocabulary, 10,000 features) for the retrieval system. Fits and transforms the entire corpus to create a search index. *(Correct: A broader vocabulary is indeed better for search retrieval than for classification.)*
*   **[22] Code:** Defines the `retrieve_answer` function to compute cosine similarity between the user query and all dataset questions, returning the top K matches. *(Correct)*
*   **[23] Code:** Tests the retrieval function across various phrasings. *(Correct)*
*   **[24] Code:** Introduces `retrieve_with_threshold` to return a fallback message if the highest similarity score is below `0.45`. *(Correct: Excellent UX addition for handling out-of-domain queries.)*
*   **[25-26] Code:** Tests the threshold logic with both irrelevant ("pizza") and relevant ("heart attack") queries. *(Correct)*
*   **[27] Code:** Saves the retrieval vectorizer, the document vectors, and the dataframe. *(Correct)*

## 4. Explainability Layer
*   **[28-29] Markdown & Code:** Introduces explainability to show *why* an answer was retrieved.
*   **[30] Code:** Defines `get_matched_keywords` to extract overlapping words between the query and the matched question, ranked by their TF-IDF weight. *(Correct)*
*   **[31] Code:** Defines `highlight_answer` which uses regex word boundaries to boldly highlight matched keywords within the retrieved answer. *(Correct: Using `\b` word boundaries prevents partial word matches.)*
*   **[32] Code:** Adds a `confidence_label` function that maps similarity scores to user-friendly labels (High, Medium, No reliable match) and emojis. *(Correct)*

## 5. Pipeline Integration
*   **[33-35] Markdown & Code:** Defines `get_explainable_answer`, combining retrieval, thresholding, highlighting, and confidence scoring into a single output dictionary, and stress-tests it. *(Correct)*
*   **[36-38] Markdown & Code:** Defines `predict_intent` using the loaded intent classifier, and wraps the entire process in a final `cliniq_pipeline` function. Tests the integrated pipeline. *(Correct: The final dictionary structure is perfect for frontend integration like Streamlit or React.)*

---

## Future Scope / Next Steps
The current pipeline is a fantastic baseline utilizing traditional Machine Learning and NLP. Here are several impactful directions to take this project to the next level:

1.  **Dense Embeddings for Semantic Search:**
    TF-IDF relies on exact word overlap. If a user searches for "hypertension" and the dataset uses "high blood pressure", TF-IDF might fail. Replacing TF-IDF with dense vector embeddings (e.g., HuggingFace `SentenceTransformers`, `BioBERT`, or `ClinicalBERT`) will enable *semantic* matching, understanding the meaning behind the words.
2.  **Generative AI / RAG (Retrieval-Augmented Generation):**
    Currently, the bot returns a static pre-written answer. You can feed the retrieved answer(s) into an LLM (like OpenAI's GPT-4, Llama 3, or Gemini) as context, and ask the LLM to synthesize a conversational, easy-to-read response based *only* on that medical context.
3.  **Advanced Intent Classification:**
    The current intent labels are created via simple rule-based keyword matching. For a production system, replacing this with a zero-shot LLM classifier or fine-tuning a small transformer model will yield much higher accuracy and allow for more nuanced intents (e.g., `dosage_query`, `side_effects_query`).
4.  **Query Expansion & Spell Correction:**
    Medical queries often contain typos (e.g., "diabetis"). Integrating a medical spell-checker or expanding queries using a medical ontology (like UMLS or SNOMED CT) before hitting the retrieval engine can significantly improve recall.
5.  **Multi-turn Conversation Context:**
    The current pipeline is stateless (single-turn). Adding memory so the user can ask follow-up questions (e.g., "What are its symptoms?", where "its" refers to the previously retrieved disease) would make the application a true chatbot.
