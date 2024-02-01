import pickle
import re
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 

app=Flask(__name__)

## Load the model
sentiment_model= pickle.load(open('Logistic_Regression_model.pkl','rb'))

## Load the Term Of Frequency Model

term_of_frequency = pickle.load(open('tfvectorizer.pkl','rb'))

## Preprocess Function

def preprocess_text(text):
    
    
    # Removendo caracteres especiais
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Removendo rastros de html (como gifs por exemplo)
    pattern = re.compile(r'<[^>]+>')
    text = re.sub(pattern, '', text)
    
    # Remover tags de spoiler entre colchetes
    text = re.sub('\[[^]]*\]', '', text)
    
    # Tokenizar o texto
    tokens = word_tokenize(text)
    
    # Remover stopwords e converter para minúsculas
    stopwords_list = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.lower() not in stopwords_list]
    
    # Aplicar lematização
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    processed_data = preprocess_text(data)
    print(processed_data)
    data_text = ' '.join(processed_data)
    print(data_text)
    data_vectorized = term_of_frequency.transform([data_text])
    output= sentiment_model.predict(data_vectorized)
    predicted_result = float(output[0])
    print(predicted_result)
    return jsonify(predicted_result)

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    processed_data = preprocess_text(user_input)
    data_text = ' '.join(processed_data)
    data_vectorized = term_of_frequency.transform([data_text])
    output = sentiment_model.predict(data_vectorized)
    predicted_result = float(output[0])

    sentiment_label = "Positive" if predicted_result == 1 else "Negative"

    return render_template('home.html', prediction_text="Sentiment Analysis Result: {}".format(sentiment_label),  predicted_result=sentiment_label)


if __name__=="__main__":
    app.run(debug=True)