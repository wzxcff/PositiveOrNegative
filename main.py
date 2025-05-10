import spacy
import pandas as pd
from keras import Sequential
from keras.src.layers import Dense, Embedding, Dropout, GlobalAveragePooling1D
from keras.src.legacy.preprocessing.text import Tokenizer
import numpy as np
from keras.src.utils import pad_sequences
import pickle

max_features = 10000

def lemmatize(text):
    res = nlp(text)
    return " ".join([tokens.lemma_ for tokens in res])

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("dataset.csv")


X = df["sentence"].apply(lemmatize)
y = df["label"]

tokenizer = Tokenizer(num_words=max_features, oov_token="OOV")
tokenizer.fit_on_texts(X)

X_sequence = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_sequence, maxlen=100)

# 10000, 16
model = Sequential([
    Embedding(max_features, 16),
    Dropout(0.2),
    GlobalAveragePooling1D(),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_padded, y, epochs=300)

# 0, 0, 1, 1, 0, 1
test_data = [
    "Waitress was a little slow in service.",
    "did not like at all.",
    "This hole in the wall has great Mexican street tacos, and friendly staff.",
    "Overall, I like this place a lot.",
    "What a shit place is that?",
    "Absolutely fell in love!"
]

model.save("positive_negative.keras")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

test_data_lemmatized = [lemmatize(text) for text in test_data]

test_sequences = tokenizer.texts_to_sequences(test_data_lemmatized)
test_padded = pad_sequences(test_sequences, maxlen=100)

predictions = model.predict(test_padded)
print(predictions)