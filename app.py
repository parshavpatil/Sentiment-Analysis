from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

# Load your trained model
model = load_model("imdb_rnn_model.h5")

# Simple tokenizer (for demo). Use your trained tokenizer for better accuracy.
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
sample_texts = ["this movie was good", "this movie was bad"]
tokenizer.fit_on_texts(sample_texts)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]

    # Convert text to sequence
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=200)

    # Predict sentiment
    prediction = model.predict(padded)[0][0]
    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"

    return render_template("index.html", review=review, sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
