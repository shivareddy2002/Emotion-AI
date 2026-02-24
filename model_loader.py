import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_emotion_assets():
    """
    Loads the H5 model, tokenizer, and label encoder from the root directory.
    Returns: (model, tokenizer, label_encoder) or (None, None, None) on failure.
    """
    try:
        from tensorflow.keras.models import load_model

        model = load_model('emotion_model.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        print('Model assets loaded successfully.')
        return model, tokenizer, label_encoder
    except Exception as e:
        print(f'Error loading model assets: {e}')
        return None, None, None
