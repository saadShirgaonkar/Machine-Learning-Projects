from flask import Flask, render_template, request

import pickle 

import pandas as pd

import numpy as np

import tensorflow as tf 

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = pickle.load(open(r"D:\Jupyter-notebook\end-2-end-deployment\Sentiment_analysis\tokenizer.pkl", "rb"))

model = load_model(r"D:\Jupyter-notebook\end-2-end-deployment\Sentiment_analysis\review.h5")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']

    df = pd.DataFrame([text], columns =['text'])

    tk = tokenizer.texts_to_sequences(df['text'].tolist())

    x_kt = pad_sequences(tk, maxlen=50)

    score = np.round(model.predict(x_kt),2)

    sentiment =  'Positive' if score > 0.5 else 'Negative'

    return render_template('result.html', sentiment = sentiment, confidence =score[0])

if __name__ == '__main__':

    app.run(port = 8000, debug=True)


    


