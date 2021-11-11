import nltk

nltk.download('popular')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import requests
from requests.auth import HTTPBasicAuth

from tensorflow.keras.models import load_model

model = load_model('model.h5')
import json
import random

intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))


def clean_up_sentence(sentence):
    # tokenizing the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stemmming each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenizing the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json, sentence):
    newword = ""
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']

    result = {}

    url = 'http://testexample.imgateway.net/api/searchall/'

    r = requests.get(url, auth=HTTPBasicAuth('username', 'password'))
    result_json = r.json()

    for p in result_json:
        for i in sentence:
            if p['herbName'].lower() == i.lower():
                newword = i
            else:
                for i in list_of_intents:
                    if i['tag'] == tag:
                        result = {"basic": random.choice(i['responses'])}

    if newword:
        inter_url = 'http://testexample.imgateway.net/api/v1/ingredients/ingredientNames/' + newword + '/223'
        # inter_url = 'http://13.238.230.40:5001/ingredients?ingredientName=' + newword
        request = requests.get(inter_url, auth=HTTPBasicAuth('username', 'password'))
        # request = requests.post(inter_url)
        med = request.text
        result = {"medical": med}

    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    sentence = clean_up_sentence(msg)
    res = getResponse(ints, intents, sentence)

    results_array = []

    if "medical" in res:
        res_json = json.loads(res["medical"])
        data = res_json["data"]

        for i in data:
            results_array.append({"drugName": i["drugName"], "recommendation": i["recommendation"]})

        final_result = {"medical": results_array}

    else:
        results_array.append(res["basic"])
        final_result = {"basic": results_array}

    return final_result


from flask import Flask
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)


class PostRequest(Resource):
    @staticmethod
    def post():
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('question', required=True)  # add args
        args = parser.parse_args()  # parse arguments to dictionary

        response = chatbot_response(args.question)
        return {'data': response}, 200  # return data with 200 OK


api.add_resource(PostRequest, '/query')  # entry point for Users
if __name__ == "__main__":
    app.run(host='0.0.0.0')
