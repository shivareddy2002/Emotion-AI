from flask import Flask, render_template, request, jsonify
import os
import re

import requests

from model_loader import load_emotion_assets

app = Flask(__name__)

# --- Runtime mode ---
# On Vercel, default to remote mode so missing TensorFlow doesn't crash the function.
DEFAULT_USE_LOCAL_MODEL = "false" if os.getenv("VERCEL") else "true"
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", DEFAULT_USE_LOCAL_MODEL).lower() == "true"
REMOTE_INFERENCE_URL = os.getenv("REMOTE_INFERENCE_URL", "").strip()
MAX_LENGTH = 40

MODEL, TOKENIZER, LABEL_ENCODER = None, None, None
pad_sequences = None
STARTUP_ERROR = None

if USE_LOCAL_MODEL:
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences as keras_pad_sequences

        MODEL, TOKENIZER, LABEL_ENCODER = load_emotion_assets()
        pad_sequences = keras_pad_sequences
    except Exception as exc:
        STARTUP_ERROR = f"Local model initialization failed: {exc}"
        USE_LOCAL_MODEL = False


def clean_text(text):
    """Standardizes input text to match the preprocessing done during training."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fallback_predict(text):
    """Lightweight keyword fallback so UI keeps working without remote/local ML service."""
    cleaned = clean_text(text)
    lexicon = {
        'joy': ['happy', 'great', 'excited', 'awesome', 'fantastic', 'smile', 'good', 'wonderful'],
        'sadness': ['sad', 'down', 'upset', 'depressed', 'cry', 'tears', 'lonely', 'hurt'],
        'anger': ['angry', 'mad', 'furious', 'annoyed', 'hate', 'rage', 'irritated'],
        'fear': ['afraid', 'fear', 'scared', 'terrified', 'anxious', 'worried', 'panic'],
        'love': ['love', 'adore', 'dear', 'sweetheart', 'care', 'romantic', 'cherish'],
        'surprise': ['wow', 'surprised', 'unexpected', 'shocked', 'amazing', 'unbelievable'],
    }

    scores = {emotion: 0 for emotion in lexicon}
    words = cleaned.split()
    for word in words:
        for emotion, terms in lexicon.items():
            if word in terms:
                scores[emotion] += 1

    if all(score == 0 for score in scores.values()):
        return {'emotion': 'neutral', 'confidence': '51.00%', 'source': 'fallback'}

    emotion = max(scores, key=scores.get)
    confidence = min(95.0, 60.0 + (scores[emotion] * 10.0))
    return {'emotion': emotion, 'confidence': f'{confidence:.2f}%', 'source': 'fallback'}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/workflow')
def workflow():
    return render_template('workflow.html')


@app.route('/health')
def health():
    status = {
        'use_local_model': USE_LOCAL_MODEL,
        'remote_inference_configured': bool(REMOTE_INFERENCE_URL),
        'startup_error': STARTUP_ERROR,
    }
    return jsonify(status), 200


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    user_text = data.get('text', '')

    if not user_text.strip():
        return jsonify({'error': 'No text provided'}), 400

    if not USE_LOCAL_MODEL:
        if not REMOTE_INFERENCE_URL:
            fallback = fallback_predict(user_text)
            fallback['warning'] = 'REMOTE_INFERENCE_URL is not configured. Showing fallback prediction.'
            return jsonify(fallback), 200

        try:
            response = requests.post(
                REMOTE_INFERENCE_URL,
                json={'text': user_text},
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()

            remote_emotion = payload.get('emotion') or payload.get('label') or payload.get('prediction')
            remote_confidence = payload.get('confidence') or payload.get('score')

            if isinstance(remote_confidence, (int, float)):
                remote_confidence = f"{float(remote_confidence) * 100:.2f}%" if float(remote_confidence) <= 1 else f"{float(remote_confidence):.2f}%"
            elif isinstance(remote_confidence, str) and '%' not in remote_confidence:
                try:
                    parsed = float(remote_confidence)
                    remote_confidence = f"{parsed * 100:.2f}%" if parsed <= 1 else f"{parsed:.2f}%"
                except ValueError:
                    remote_confidence = None

            if remote_emotion and remote_confidence:
                return jsonify({
                    'emotion': str(remote_emotion),
                    'confidence': str(remote_confidence),
                    'source': 'remote',
                }), 200

            fallback = fallback_predict(user_text)
            fallback['warning'] = 'Remote API returned an invalid payload. Showing fallback prediction.'
            return jsonify(fallback), 200
        except (requests.RequestException, ValueError) as exc:
            fallback = fallback_predict(user_text)
            fallback['warning'] = f'Remote inference failed, fallback used: {exc}'
            return jsonify(fallback), 200

    if MODEL is None or pad_sequences is None:
        fallback = fallback_predict(user_text)
        fallback['warning'] = f'Model not loaded, fallback used: {STARTUP_ERROR}'
        return jsonify(fallback), 200

    cleaned_text = clean_text(user_text)
    sequence = TOKENIZER.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')
    prediction_scores = MODEL.predict(padded_sequence, verbose=0)

    class_index = prediction_scores.argmax()
    emotion = LABEL_ENCODER.inverse_transform([class_index])[0]
    confidence = float(prediction_scores.max())

    return jsonify({'emotion': emotion, 'confidence': f"{confidence * 100:.2f}%", 'source': 'model'})


if __name__ == '__main__':
    app.run(debug=True)
