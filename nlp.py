import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = [
    # SAD
    "i feel very sad",
    "everything feels heavy",
    "naaku chala badha ga undi",
    "nenu lonely ga unna",
    "Naakuuu nuvvuu istammm leduuuuu",
    "Naakuuu nuvvuu istammm ledd",
    "Naku nuvvu istam led",


    # HAPPY
    "i am feeling great today",
    "this made me really happy",
    "naaku chala happy ga undi",
    "ee roju chala bagundi",
    "I love you ❤️",
    "Bangarammmmmmmmm",
    "Naakuuu nuvvuu istammm",
    "My Bangarammmmm",
    "Naa Babyyyy",
    "My Babyyyyyy",
    "Myyyy Babyyyyyyyy",

    # MISSING YOU
    "i miss you so much",
    "i wish you were here",
    "nenu ninnu chala miss avutunnanu",
    "manam kalisi undali ani undi",

    # ANGRY
    "i am really angry",
    "this hurt me a lot",
    "naaku kopam vastondi",
    "nuvvu ala cheppakudadhu",
    "Dengey eheeee ",

    # CALM
    "i feel peaceful",
    "everything is okay",
    "naaku calm ga undi",
    "just relaxing now"
]

labels = [
    0, 0, 0, 0,0,0,0,   # sad
    1, 1, 1, 1,1,1,1,1,1,1,1,   # happy
    2, 2, 2, 2,   # missing
    3, 3, 3, 3,3,   # angry
    4, 4, 4, 4    # calm
]

VOCAB_SIZE = 5000
MAX_LEN = 30
OOV_TOKEN = "<OOV>"

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

x = np.array(padded)
y = np.array(labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32, input_length=MAX_LEN),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # 5 moods
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x,y,epochs=500, verbose=1)

mood_map = {
    0: "Sad",
    1: "Happy",
    2: "Missing You",
    3: "Angry",
    4: "Calm",
}

import pickle

model.save("mood_model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("word_index.pkl", "wb") as f:
    pickle.dump(tokenizer.word_index, f)
    
def predict_mood(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    preds = model.predict(pad)
    confidence = np.max(preds)
    mood_id = np.argmax(preds)

    if confidence < 0.6:
        return "MIXED FEELINGS", confidence
    return mood_map[mood_id], confidence

mood, confidence = predict_mood("Bangarammm ❤️")
print(mood, confidence)


