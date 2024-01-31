import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

## Load the model
sentiment_model= pickle.load(open('Logistic_Regression_model.pkl','rb'))

# Preprocess Function

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
    new_data = preprocess_text(data)
    output= sentiment_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)