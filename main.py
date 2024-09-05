from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import nltk

# Initialize NLTK
nltk.download('punkt')

class Scorer:
    def __init__(self, docs):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.doc_tfidf = self.vectorizer.fit_transform(docs)
        self.features = [self._feature_tfidf, self._feature_positive_feedback]
        self.feature_weights = [1., 2.]
        self.feedback = {}

    def score(self, query):
        feature_vectors = [feature(query) for feature in self.features]
        feature_vectors_weighted = [feature * weight for feature, weight in zip(feature_vectors, self.feature_weights)]
        return np.sum(feature_vectors_weighted, axis=0)

    def learn_feedback(self, feedback_dict):
        self.feedback = feedback_dict

    def _feature_tfidf(self, query):
        query_vector = self.vectorizer.transform([query])
        similarity = cosine_similarity(query_vector, self.doc_tfidf)
        return similarity.ravel()

    def _feature_positive_feedback(self, query):
        if not self.feedback:
            return np.zeros(len(self.docs))
        feedback_queries = list(self.feedback.keys())
        similarity = cosine_similarity(self.vectorizer.transform([query]), self.vectorizer.transform(feedback_queries))
        nn_similarity = np.max(similarity)
        nn_idx = np.argmax(similarity)
        pos_feedback_doc_idx = [idx for idx, feedback_value in self.feedback[feedback_queries[nn_idx]] if feedback_value == 1.]
        feature_values = {doc_idx: nn_similarity * count / sum(counts.values()) for doc_idx, count in Counter(pos_feedback_doc_idx).items()}
        return np.array([feature_values.get(doc_idx, 0.) for doc_idx, _ in enumerate(self.docs)])

# Example documents
docs = [
    "About us. We deliver Artificial Intelligence & Machine Learning solutions to solve business challenges.",
    "Contact information. Email [martin davtyan at filament dot ai] if you have any questions.",
    "Filament Chat. A framework for building and maintaining a scalable chatbot capability."
]

scorer = Scorer(docs)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/score', methods=['POST'])
def score_query():
    data = request.json
    query = data.get('query', '')
    score = scorer.score(query).tolist()
    return jsonify({'scores': score})

@app.route('/feedback', methods=['POST'])
def provide_feedback():
    data = request.json
    feedback = data.get('feedback', {})
    scorer.learn_feedback(feedback)
    return jsonify({'message': 'Feedback received'})

if __name__ == '__main__':
    app.run(debug=True)
