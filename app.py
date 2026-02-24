from flask import Flask, render_template, request, jsonify
import os
import re
import requests

from model_loader import load_emotion_assets

app = Flask(__name__)

# --- Runtime mode ---
# In Vercel, set USE_LOCAL_MODEL=false and provide REMOTE_INFERENCE_URL to avoid heavy ML dependencies.
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "true").lower() == "true"
REMOTE_INFERENCE_URL = os.getenv("REMOTE_INFERENCE_URL", "").strip()
MAX_LENGTH = 40

MODEL, TOKENIZER, LABEL_ENCODER = None, None, None
pad_sequences = None

if USE_LOCAL_MODEL:
    from tensorflow.keras.preprocessing.sequence import pad_sequences as keras_pad_sequences

    MODEL, TOKENIZER, LABEL_ENCODER = load_emotion_assets()
    pad_sequences = keras_pad_sequences


def clean_text(text):
    """Standardizes input text to match the preprocessing done during training."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# --- Page Routes ---
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/workflow')
def workflow():
    return render_template('workflow.html')


# --- API Route for Predictions ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    user_text = data.get('text', '')

    if not user_text.strip():
        return jsonify({'error': 'No text provided'}), 400

    if not USE_LOCAL_MODEL:
        if not REMOTE_INFERENCE_URL:
            return jsonify({'error': 'REMOTE_INFERENCE_URL is not configured'}), 500

        try:
            response = requests.post(
                REMOTE_INFERENCE_URL,
                json={'text': user_text},
                timeout=20,
            )
            return jsonify(response.json()), response.status_code
        except requests.RequestException as exc:
            return jsonify({'error': f'Remote inference failed: {exc}'}), 502

    if MODEL is None:
        return jsonify({'error': 'Model not loaded on server'}), 500

    cleaned_text = clean_text(user_text)
    sequence = TOKENIZER.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')
    prediction_scores = MODEL.predict(padded_sequence, verbose=0)

    class_index = prediction_scores.argmax()
    emotion = LABEL_ENCODER.inverse_transform([class_index])[0]
    confidence = float(prediction_scores.max())

    return jsonify({'emotion': emotion, 'confidence': f"{confidence * 100:.2f}%"})


if __name__ == '__main__':
    app.run(debug=True)
