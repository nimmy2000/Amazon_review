from flask import Flask, render_template, request
import joblib
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# =========================
# Load Traditional ML Models
# =========================
logistic_model = joblib.load("Review/logistic_model.pkl")
naive_bayes_model = joblib.load("Review/naive_bayes_model.pkl")
svm_model = joblib.load("Review/svm_model.pkl")
rf_model = joblib.load("Review/rf_model.pkl")   

# =========================
# Load Deep Learning Models
# =========================
rnn_model = load_model("Review/rnn_model.keras")
lstm_model = load_model("Review/lstm_model.keras")


# =========================
# Load Vectorizer & Tokenizer
# =========================
vectorizer = joblib.load("Review/tfidf_vectorizer.pkl")
with open("Review/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_SEQUENCE_LENGTH = 100  # Should match training

# =========================
# Initialize Flask App
# =========================
app = Flask(__name__)

# =========================
# Home Route
# =========================
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    algorithm_name = None

    if request.method == 'POST':
        review = request.form['review']
        selected_algo = request.form['algorithm']

        # ----- Traditional ML Models -----
        if selected_algo in ['logistic', 'naive_bayes', 'svm','random_forest']:
            review_features = vectorizer.transform([review])
            if selected_algo == 'logistic':
                pred_label = logistic_model.predict(review_features)[0]
                algorithm_name = "Logistic Regression"
            elif selected_algo == 'naive_bayes':
                pred_label = naive_bayes_model.predict(review_features)[0]
                algorithm_name = "Naive Bayes"
            elif selected_algo == 'random_forest':
                pred_label = rf_model.predict(review_features)[0]
                algorithm_name = "Random Forest"
            elif selected_algo == 'svm':
                pred_label = svm_model.predict(review_features)[0]
                algorithm_name = "Support Vector Machine"

            prediction = "Positive" if pred_label == 1 else "Negative"
   

        # ----- Deep Learning Models (RNN/LSTM) -----
        elif selected_algo in ['rnn', 'lstm']:
            sequences = tokenizer.texts_to_sequences([review])
            padded_seq = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

            if selected_algo == 'rnn':
                pred_prob = rnn_model.predict(padded_seq)[0][0]
                algorithm_name = "RNN"
            elif selected_algo == 'lstm':
                pred_prob = lstm_model.predict(padded_seq)[0][0]
                algorithm_name = "LSTM"

            # Convert probability to label
            pred_label = int(pred_prob >= 0.5)
            prediction = f"Positive ({pred_prob:.2f})" if pred_label == 1 else f"Negative ({1 - pred_prob:.2f})"

    return render_template('index.html', prediction=prediction, algorithm_name=algorithm_name)

# =========================
# Run App
# =========================
if __name__ == '__main__':
    app.run(debug=True)
