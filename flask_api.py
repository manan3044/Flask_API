import os
import re
import string
import tempfile
import pickle
import whisper

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import nltk
import cv2
import pytesseract
import ffmpeg

import torch
import torchaudio
import os
from RawNetLite import RawNetLite
from audio_preprocessor import preprocess_audio

import numpy as np
import torch.nn as nn

# --- SETUP ---
# Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\\tesseract.exe"

# Download stopwords if not already done
nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

# Load models
with open("my_models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("my_models/phishing_model2.pkl", "rb") as f:
    phising_model = pickle.load(f)

# # Audio model paths
# MODEL_PATH = "models/model_logical_CCE_100_32_0.0001/model_best_epoch100.pth.tar"
# CONFIG_PATH = "config/model_config_RawNet.yaml"

# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
# if not os.path.exists(CONFIG_PATH):
#     raise FileNotFoundError(f"Config not found at {CONFIG_PATH}")

# from audio_spoof_detector import analyze_audio_file

model_transcribe = whisper.load_model("base") 


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, lstm_units=64, dropout_rate=0.5):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, lstm_units, num_layers=1, 
                            bidirectional=True, batch_first=True)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_units * 2, 32)
        self.dropout2 = nn.Dropout(dropout_rate / 2)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = self.global_max_pool(x).squeeze(-1)
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc2(x)).squeeze(-1)
        return x

vocab_size = 20000  
model = BiLSTMClassifier(vocab_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('my_models/phishing_text_model.pth', map_location=device))
model.to(device)
model.eval()

class TokenizerCustom:
    def __init__(self, word_index, max_len=300):
        self.word_index = word_index
        self.oov_token = '<OOV>'
        self.oov_index = 1
        self.max_len = max_len

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = [self.word_index.get(word, self.oov_index) for word in text.split()]
            sequences.append(seq)
        return sequences

    def pad_sequences(self, sequences):
        padded = np.zeros((len(sequences), self.max_len), dtype=int)
        for i, seq in enumerate(sequences):
            trunc = seq[:self.max_len]
            padded[i, -len(trunc):] = trunc  # right align
        return padded

with open("my_models/tokenizer_text.pkl", "rb") as f:   # <-- path to your saved tokenizer.pkl
    keras_tokenizer = pickle.load(f)

tokenizer_word_index = keras_tokenizer.word_index
tokenizer = TokenizerCustom(tokenizer_word_index, max_len=300)

# --- FLASK APP ---
app = Flask(__name__)

# ---------------- UTILS ----------------
def preprocess_text_email(text: str) -> str:
    """Clean and preprocess input text."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

def preprocess_text_text(text):
    text = str(text).lower()
    text = re.sub('<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def run_audio_analysis(file_path: str) -> dict:
    """Run audio spoof detection and format result."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawNetLite().to(device)
    model.load_state_dict(torch.load("my_models/cross_domain_rawnet_lite.pt", map_location=device))
    model.eval()
    
    # Load and process audio
    waveform, sr = torchaudio.load(file_path)
    audio_tensor = preprocess_audio(waveform, sr, target_sr=16000, target_sec=3.0)
    audio_tensor = audio_tensor.unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(audio_tensor)
        probability = output.item()
        
        if probability > 0.5:
            result = 1
            confidence = probability
        else:
            result = 0
            confidence = 1 - probability
        
        return result, confidence


def extract_audio_from_video(video_path: str) -> str:
    """Extract audio from video using ffmpeg."""
    tmp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_audio.close()
    ffmpeg.input(video_path).output(
        tmp_audio.name, format='wav', acodec='pcm_s16le', ac=1, ar='16000'
    ).run(quiet=True, overwrite_output=True)
    return tmp_audio.name


def image_ocr(image_path: str) -> str:
    """Extract text from image using pytesseract."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    config = "-l eng --oem 3 --psm 6"
    return pytesseract.image_to_string(thresh, config=config)

def transcribe(video_path: str) -> str:
    result = model_transcribe.transcribe(video_path)
    return result

def predict_email_func(text):
    clean_text = preprocess_text_text(text)
    seq = tokenizer.texts_to_sequences([clean_text])
    pad_seq = tokenizer.pad_sequences(seq)
    input_tensor = torch.tensor(pad_seq, dtype=torch.long).to(device)

    with torch.no_grad():
        prob = model(input_tensor).item()
    
    label = "Phishing" if prob > 0.5 else "Safe"
    return label, prob


# ---------------- ROUTES ----------------
# Text prediction
# @app.route("/predict-email", methods=["POST"])
# def predict_email_route():
#     data = request.get_json()
#     if not data or "text" not in data:
#         return jsonify({"error": "Missing 'text' field"}), 400

#     try:
#         cleaned_text = preprocess_text_email(data["text"])
#         text_tfidf = tfidf.transform([cleaned_text])
#         prediction = int(phising_model.predict(text_tfidf)[0])
#         probability = phising_model.predict_proba(text_tfidf).tolist()

#         return jsonify({
#             "cleaned_text": cleaned_text,
#             "prediction": prediction,
#             "probability": probability
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
@app.route("/predict-text", methods=["POST"])
def predict_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    try:
        label, prob = predict_email_func(data["text"])
        print(label)
        label = 1 if label == "Phishing" else 0

        return jsonify({
            "cleaned_text": "",
            "prediction": label,
            "probability": prob
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Video prediction
@app.route("/predict-video", methods=["POST"])
def predict_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded. Use key 'video'"}), 400

    video_file = request.files["video"]
    if not video_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    video_path = None
    audio_path = None
    try:
        filename = secure_filename(video_file.filename)
        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
        video_file.save(tmp_video.name)
        video_path = tmp_video.name
        tmp_video.close()

        audio_path = extract_audio_from_video(video_path)

        transcribed_text = transcribe(audio_path)
        print(transcribed_text['text'])
        label, prob = predict_email_func(transcribed_text)
        print(label)
        label = 1 if label == "Phishing" else 0

        result, confidence = run_audio_analysis(audio_path)

        return jsonify({
            "prediction": label,
            "confidence": float(confidence),
            "message": "Bonafide ✅" if label == 0 else "Spoofed ❌",
            "is_spoofed": "authentic" if result == 0 else "spoofed"
        })

    except Exception as e:
        print("Error in /predict-video:", e)
        return jsonify({"error": str(e)}), 500

    finally:
        for f in (video_path, audio_path):
            if f and os.path.exists(f):
                os.remove(f)


# Image prediction
@app.route("/predict-image", methods=["POST"])
def predict_text_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded. Use key 'image'"}), 400

    image_file = request.files["image"]
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image_file.save(tmp.name)
            save_path = tmp.name
        print(save_path)

        extracted_text = image_ocr(save_path)
        label, prob = predict_email_func(extracted_text)
        label = 1 if label == "Phishing" else 0

        return jsonify({
            "cleaned_text": "",
            "prediction": label,
            "probability": prob
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if save_path and os.path.exists(save_path):
            os.remove(save_path)

# Audio prediction
@app.route("/predict-audio", methods=["POST"])
def predict_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded. Use key 'audio'"}), 400

    audio_file = request.files["audio"]
    if not audio_file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        filename = secure_filename(audio_file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
            audio_file.save(tmp.name)
            temp_path = tmp.name

        transcribed_text = transcribe(temp_path)
        print(transcribed_text['text'])
        label, prob = predict_email_func(transcribed_text)
        print(label)
        label = 1 if label == "Phishing" else 0
        
        result = run_audio_analysis(temp_path)
        result, confidence = run_audio_analysis(temp_path)

        return jsonify({
            "prediction": label,
            "confidence": float(confidence),
            "message": "Bonafide ✅" if label == 0 else "Spoofed ❌",
            "is_spoofed": "authentic" if result == 0 else "spoofed"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=False, port=5000)
