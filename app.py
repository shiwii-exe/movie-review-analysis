from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer
model = tf.keras.models.load_model("sentiment_model.keras")
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Configuration
MAX_SEQUENCE_LENGTH = 200  # Make sure this matches what you used during training

# Initialize Flask app
app = Flask(__name__)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    review = data.get("review", "")

    # Preprocess the review
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

    # Predict sentiment
    prediction = model.predict(padded_sequence)[0][0]

    # If prediction > 0.5, it's positive; else negative
    sentiment = "Positive" if prediction > 0.5 else "Negative"

    # Optionally: Print to console for debugging
    print(f"Review: {review}")
    print(f"Prediction score: {prediction}")
    print(f"Predicted Sentiment: {sentiment}")

    # Return response as JSON
    return jsonify({
        'sentiment': sentiment,
        'score': float(prediction)  # optional, you can use this in frontend
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
