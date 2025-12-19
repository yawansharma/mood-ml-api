from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# --------- HELPERS ---------
def normalize_text(text):
    text = text.lower()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # cooool → cool
    return text

# --------- LOAD MODEL ---------
model = tf.keras.models.load_model("mood_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 30

mood_map = {
    0: "Sad",
    1: "Happy",
    2: "Missing You",
    3: "Angry",
    4: "Calm",
}

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    text = normalize_text(text)

    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

    preds = model.predict(pad)
    confidence = float(np.max(preds))
    mood_id = int(np.argmax(preds))

    if confidence < 0.7:
        return jsonify({
            "mood": "Mixed Feelings",
            "confidence": confidence
        })

    return jsonify({
        "mood": mood_map[mood_id],
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

