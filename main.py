import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

with open("intents.json") as file:
    data = json.load(file)

# print(data)

words = []
labels = []
docs = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # tokenize each word
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        # add to documents in our corpus
        docs.append(pattern)
        # add to labels if not already there
    if intent["tag"] not in labels:
        labels.append(intent["tag"])
