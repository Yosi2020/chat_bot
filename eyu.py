import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np 
import tflearn as tl 
import tensorflow as tf
import json
import pickle

print("I am starting load data")

with open(r'C:\Users\Eyosiyas\Desktop\chatbot\chat.json') as file:
    data = json.load(file)

print("I finshed my loading json file")

try:
    print("I am starting loading pickle file")
    with open(r"C:\Users\Eyosiyas\Desktop\chatbot\data.pickle", "rb") as f:
        words, lables, trainning, output = pickle.load(f)
    print(labels[0])
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)
    trainning = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else: bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] =1

        trainning.append(bag)
        output.append(output_row) 

    trainning = np.array(trainning)
    output = np.array(output)

    with open(r"C:\Users\Eyosiyas\Desktop\chatbot\data.pickle", "wb") as f:
        pickle.dump((words, labels, trainning, output), f)

from tensorflow.python.framework import ops
ops.reset_default_graph() 

net = tl.input_data(shape = [None, len(trainning[0])])
net = tl.fully_connected(net, 8)
net = tl.fully_connected(net, 8)
net = tl.fully_connected(net, len(output[0]), activation ="softmax")
net = tl.regression(net)

model = tl.DNN(net)

try:
    print("I am loading the model")
    model.load(r"C:\Users\Eyosiyas\Desktop\chatbot\model.tflearn")
except:
    model.fit(trainning, output, n_epoch=1000, batch_size = 8, show_metric = True)
    model.save(r"C:\Users\Eyosiyas\Desktop\chatbot\model.tflearn")

def bag_of_word(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(words.lower()) for words in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


import random
import datetime
import webbrowser
import pyttsx3
import wikipedia
from pygame import mixer
import speech_recognition as sr
from speech_recognition.__main__ import r, audio

def chat(inp):
    print("I am starting ...")
    #while True:
        #inp = input("You: ")
    results = model.predict([bag_of_word(inp, words)])
    results_index = np.argmax(results)
    tag = labels[results_index]

    if results_index > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))
    else: 
        print("I didn't understand your questions")


def record():
    print("start talking with the bot (type q to stop)")
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    volume = engine.getProperty('volume')
    engine.setProperty('volume', 10.0)
    rate = engine.getProperty('rate')

    engine.setProperty('rate', rate - 25)
    while True:
        now = datetime.datetime.now()
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Tell me something:")
            audio = r.listen(source)
            try:
                print("You said:- " + r.recognize_google(audio))
                input_a = r.recognize_google(audio)
                chat(input_a)
            except sr.UnknownValueError:
                    print("Could not understand audio")
                    engine.say('I didnt get that. Rerun the code')

                    engine.runAndWait()

record()
