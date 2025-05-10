from keras import models
import spacy
from keras.src.utils import pad_sequences
import pickle
import json

json_list = []


nlp = spacy.load("en_core_web_sm")

def lemmatize(text):
    res = nlp(text)
    return " ".join([tokens.lemma_ for tokens in res])


model = models.load_model("positive_negative.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


while True:
    user_input = input("Please enter a sentence to analyze: ")
    if user_input == "exit": break
    user_sequence = tokenizer.texts_to_sequences([lemmatize(user_input)])
    user_padded = pad_sequences(user_sequence, maxlen=100)
    prediction = model.predict(user_padded, verbose=0)
    prediction_rounded = round(prediction[0][0])

    result = "positive" if prediction_rounded == 1 else "negative"

    print(f"Model predicted that this text is {result}")

    json_list.append({"user_input": user_input, "prediction": result})

with open("positive_negative.json", "w") as f:
    json.dump(json_list, f, indent=4)