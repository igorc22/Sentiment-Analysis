import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

## Load the model
sentiment_model=pickle.load(open('.pkl','rb'))
preprocess_function = pickle.load(open(' .pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',method=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    new_data = preprocess_function.preprocess(data)
    output= sentiment_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)