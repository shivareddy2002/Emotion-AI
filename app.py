from flask import Flask, render_template, request, jsonify
import os
import re

import requests

from model_loader import load_emotion_assets

app = Flask(__name__)

# --- Runtime mode ---
# Keep Vercel lightweight by default. Use local model only when explicitly enabled.
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
REMOTE_INFERENCE_URL = os.getenv("REMOTE_INFERENCE_URL", "").strip()
MAX_LENGTH = 40

MODEL, TOKENIZER, LABEL_ENCODER = None, None, None
pad_sequences = None
STARTUP_ERROR = None
MODEL_INIT_ATTEMPTED = False


def ensure_local_model_initialized():
    """Lazy-load local model dependencies only when needed."""
    global MODEL, TOKENIZER, LABEL_ENCODER, pad_sequences, STARTUP_ERROR, MODEL_INIT_ATTEMPTED

    if MODEL_INIT_ATTEMPTED:
        return MODEL is not None and pad_sequences is not None

    MODEL_INIT_ATTEMPTED = True
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences as keras_pad_sequences

        MODEL, TOKENIZER, LABEL_ENCODER = load_emotion_assets()
        pad_sequences = keras_pad_sequences
        if MODEL is None:
            STARTUP_ERROR = "Model files could not be loaded."
            return False
        return True
    except Exception as exc:
        STARTUP_ERROR = f"Local model initialization failed: {exc}"
        return False


def clean_text(text):
    """Standardizes input text to match the preprocessing done during training."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fallback_predict(text):
    """Lightweight lexicon fallback with phrase boosts and negation handling."""
    cleaned = clean_text(text)
    words = cleaned.split()

    lexicon = {
        'joy': {'happy': 1.4, 'great': 1.2, 'excited': 1.4, 'awesome': 1.4, 'fantastic': 1.5, 'smile': 1.0, 'good': 0.8, 'wonderful': 1.5, 'glad': 1.0},
        'sadness': {'sad': 1.4, 'empty': 1.4, 'pointless': 1.5, 'down': 1.0, 'upset': 1.2, 'depressed': 1.6, 'cry': 1.2, 'tears': 1.2, 'lonely': 1.4, 'hurt': 1.1},
        'anger': {'angry': 1.5, 'mad': 1.2, 'furious': 1.6, 'annoyed': 1.1, 'hate': 1.5, 'rage': 1.5, 'irritated': 1.2},
        'fear': {'afraid': 1.4, 'fear': 1.2, 'scared': 1.5, 'terrified': 1.6, 'anxious': 1.3, 'worried': 1.1, 'panic': 1.5},
        'love': {'love': 1.7, 'adore': 1.7, 'dear': 1.0, 'sweetheart': 1.3, 'care': 1.1, 'romantic': 1.4, 'cherish': 1.5},
        'surprise': {'wow': 1.3, 'surprised': 1.4, 'unexpected': 1.4, 'shocked': 1.5, 'amazing': 1.2, 'unbelievable': 1.4},
    }

    phrase_boosts = {
        'love': ['i love', 'love you', 'in love', 'miss you'],
        'sadness': ['feel empty', 'pointless and empty', 'feel hopeless', 'broken inside'],
        'joy': ['so happy', 'very happy', 'feeling great'],
        'anger': ['so angry', 'very angry', 'makes me mad'],
        'fear': ['so scared', 'very scared', 'panic attack'],
        'surprise': ['did not expect', 'never expected', 'what a surprise'],
    }

    negations = {'not', 'never', 'no', "can't", "dont", "don't"}
    scores = {emotion: 0.0 for emotion in lexicon}

    for idx, word in enumerate(words):
        prev = words[idx - 1] if idx > 0 else ''
        negated = prev in negations
        for emotion, terms in lexicon.items():
            weight = terms.get(word)
            if weight:
                scores[emotion] += (-0.6 * weight) if negated else weight

    for emotion, phrases in phrase_boosts.items():
        for phrase in phrases:
            if phrase in cleaned:
                scores[emotion] += 1.8

    best_emotion = max(scores, key=scores.get)
    best_score = scores[best_emotion]
    if best_score <= 0:
        return {'emotion': 'neutral', 'confidence': '55.00%', 'source': 'lite-model'}

    total_positive = sum(max(0.0, s) for s in scores.values())
    confidence = 55.0 if total_positive == 0 else min(96.0, 55.0 + (best_score / total_positive) * 40.0)
    return {'emotion': best_emotion, 'confidence': f'{confidence:.2f}%', 'source': 'lite-model'}


def normalize_remote_payload(payload):
    remote_emotion = payload.get('emotion') or payload.get('label') or payload.get('prediction')
    remote_confidence = payload.get('confidence') or payload.get('score')

    if isinstance(remote_confidence, (int, float)):
        value = float(remote_confidence)
        remote_confidence = f"{value * 100:.2f}%" if value <= 1 else f"{value:.2f}%"
    elif isinstance(remote_confidence, str) and '%' not in remote_confidence:
        try:
            parsed = float(remote_confidence)
            remote_confidence = f"{parsed * 100:.2f}%" if parsed <= 1 else f"{parsed:.2f}%"
        except ValueError:
            remote_confidence = None

    if remote_emotion and remote_confidence:
        return {
            'emotion': str(remote_emotion),
            'confidence': str(remote_confidence),
            'source': 'remote',
        }
    return None


def fallback_predict(text):
    """Lightweight lexicon fallback with phrase boosts and negation handling."""
    cleaned = clean_text(text)
    words = cleaned.split()

    lexicon = {
        'joy': {'happy': 1.4, 'great': 1.2, 'excited': 1.4, 'awesome': 1.4, 'fantastic': 1.5, 'smile': 1.0, 'good': 0.8, 'wonderful': 1.5, 'glad': 1.0},
        'sadness': {'sad': 1.4, 'empty': 1.4, 'pointless': 1.5, 'down': 1.0, 'upset': 1.2, 'depressed': 1.6, 'cry': 1.2, 'tears': 1.2, 'lonely': 1.4, 'hurt': 1.1},
        'anger': {'angry': 1.5, 'mad': 1.2, 'furious': 1.6, 'annoyed': 1.1, 'hate': 1.5, 'rage': 1.5, 'irritated': 1.2},
        'fear': {'afraid': 1.4, 'fear': 1.2, 'scared': 1.5, 'terrified': 1.6, 'anxious': 1.3, 'worried': 1.1, 'panic': 1.5},
        'love': {'love': 1.7, 'adore': 1.7, 'dear': 1.0, 'sweetheart': 1.3, 'care': 1.1, 'romantic': 1.4, 'cherish': 1.5},
        'surprise': {'wow': 1.3, 'surprised': 1.4, 'unexpected': 1.4, 'shocked': 1.5, 'amazing': 1.2, 'unbelievable': 1.4},
    }

    phrase_boosts = {
        'love': ['i love', 'love you', 'in love', 'miss you'],
        'sadness': ['feel empty', 'pointless and empty', 'feel hopeless', 'broken inside'],
        'joy': ['so happy', 'very happy', 'feeling great'],
        'anger': ['so angry', 'very angry', 'makes me mad'],
        'fear': ['so scared', 'very scared', 'panic attack'],
        'surprise': ['did not expect', 'never expected', 'what a surprise'],
    }

    negations = {'not', 'never', 'no', "can't", "dont", "don't"}
    scores = {emotion: 0.0 for emotion in lexicon}

    for idx, word in enumerate(words):
        prev = words[idx - 1] if idx > 0 else ''
        negated = prev in negations
        for emotion, terms in lexicon.items():
            weight = terms.get(word)
            if weight:
                scores[emotion] += (-0.6 * weight) if negated else weight

    for emotion, phrases in phrase_boosts.items():
        for phrase in phrases:
            if phrase in cleaned:
                scores[emotion] += 1.8

    best_emotion = max(scores, key=scores.get)
    best_score = scores[best_emotion]
    if best_score <= 0:
        return {'emotion': 'neutral', 'confidence': '55.00%', 'source': 'lite-model'}

    total_positive = sum(max(0.0, s) for s in scores.values())
    confidence = 55.0 if total_positive == 0 else min(96.0, 55.0 + (best_score / total_positive) * 40.0)
    return {'emotion': best_emotion, 'confidence': f'{confidence:.2f}%', 'source': 'lite-model'}


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


@app.route('/health')
def health():
    return jsonify({
        'use_local_model': USE_LOCAL_MODEL,
        'remote_inference_configured': bool(REMOTE_INFERENCE_URL),
        'local_model_initialized': MODEL_INIT_ATTEMPTED and MODEL is not None,
        'startup_error': STARTUP_ERROR,
    }), 200


@app.route('/health')
def health():
    status = {
        'use_local_model': USE_LOCAL_MODEL,
        'remote_inference_configured': bool(REMOTE_INFERENCE_URL),
        'startup_error': STARTUP_ERROR,
    }
    return jsonify(status), 200


@app.route('/health')
def health():
    status = {
        'use_local_model': USE_LOCAL_MODEL,
        'remote_inference_configured': bool(REMOTE_INFERENCE_URL),
        'startup_error': STARTUP_ERROR,
    }
    return jsonify(status), 200


# --- API Route for Predictions ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    user_text = data.get('text', '')

    if not user_text.strip():
        return jsonify({'error': 'No text provided'}), 400

    # Prefer true model predictions when local model mode is explicitly enabled and available.
    if USE_LOCAL_MODEL and ensure_local_model_initialized():
        cleaned_text = clean_text(user_text)
        sequence = TOKENIZER.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')
        prediction_scores = MODEL.predict(padded_sequence, verbose=0)

        class_index = prediction_scores.argmax()
        emotion = LABEL_ENCODER.inverse_transform([class_index])[0]
        confidence = float(prediction_scores.max())

        return jsonify({'emotion': emotion, 'confidence': f"{confidence * 100:.2f}%", 'source': 'model'})

    if REMOTE_INFERENCE_URL:
        try:
            response = requests.post(REMOTE_INFERENCE_URL, json={'text': user_text}, timeout=20)
            response.raise_for_status()
            payload = response.json()
            normalized = normalize_remote_payload(payload)
            if normalized:
                return jsonify(normalized), 200

            fallback = fallback_predict(user_text)
            fallback['warning'] = 'Remote API returned invalid prediction format. Using built-in lite model.'
            return jsonify(fallback), 200
        except (requests.RequestException, ValueError) as exc:
            fallback = fallback_predict(user_text)
            fallback['warning'] = f'Remote inference failed. Using built-in lite model. ({exc})'
            return jsonify(fallback), 200

    fallback = fallback_predict(user_text)
    if USE_LOCAL_MODEL and STARTUP_ERROR:
        fallback['warning'] = f'Local model unavailable. Using built-in lite model. ({STARTUP_ERROR})'
    return jsonify(fallback), 200


if __name__ == '__main__':
    app.run(debug=True)
