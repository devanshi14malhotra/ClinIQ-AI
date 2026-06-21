# ClinIQ AI - pipeline.py
# Final combined pipeline: intent prediction + retrieval + explainability

from vectorizer import load_intent_components, predict_intent
from retrieval import load_retrieval_components, retrieve_answer
from explainability import get_explainable_answer


class ClinIQPipeline:
    """
    Loads all models once at startup, exposes a single .ask() method.
    Wrapping in a class avoids reloading .pkl files on every query —
    Streamlit will instantiate this once and reuse it across the session.
    """

    def __init__(self, threshold=0.45):
        self.threshold = threshold

        # load intent classifier components
        self.intent_vectorizer, self.intent_model = load_intent_components()

        # load retrieval components
        self.retrieval_vectorizer, self.question_vectors, self.df = load_retrieval_components()

    def ask(self, user_query):
        if not user_query or not user_query.strip():
            return {'status': 'invalid_input', 'message': "Please enter a question."}

        predicted_intent = predict_intent(user_query, self.intent_vectorizer, self.intent_model)

        response = get_explainable_answer(
            user_query,
            self.retrieval_vectorizer,
            self.question_vectors,
            self.df,
            retrieve_answer,        # passed in here, resolves the circular import issue
            threshold=self.threshold
        )
        response['predicted_intent'] = predicted_intent

        return response


if __name__ == '__main__':
    # quick manual test: python pipeline.py
    cliniq = ClinIQPipeline()

    test_queries = [
        "what are the symptoms of diabetes",
        "asdkjasdkj random gibberish",
        "how is asthma treated"
    ]

    for q in test_queries:
        print(f"\nQUERY: {q}")
        output = cliniq.ask(q)
        print("STATUS:", output['status'])
        print("PREDICTED INTENT:", output.get('predicted_intent'))
        if output['status'] == 'match_found':
            print("CONFIDENCE:", output['confidence_emoji'], output['confidence_label'])
            print("MATCHED Q:", output['matched_question'])