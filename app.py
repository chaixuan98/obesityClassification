from cgitb import text
import string
from flask import Flask, jsonify, render_template, request
import pickle

import numpy as np
from sklearn import metrics 
app = Flask(__name__)
model = pickle.load(open('model.pkl','rb')) #read mode
@app.route("/")
def home():
    return render_template('index.html')
@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        #access the data from form
        ## Age
        Gender = int(request.form["Gender"])
        Age = int(request.form["Age"])
        Height = float(request.form["Height"])
        Weight = float(request.form["Weight"])
        Family = int(request.form["Family"])
        TDEE = int(request.form["TDEE"])
        Smoker = int(request.form["Smoker"])
        Water = int(request.form["Water"])
        ActivityLevel = int(request.form["ActivityLevel"])
        Alcohol = int(request.form["Alcohol"])

        #get prediction
        input_cols = np.array([[Gender, Age, Height, Weight, Family,TDEE,Smoker,Water,ActivityLevel,Alcohol]])
        print(input_cols)
        prediction = model.predict(input_cols)[0]
        print(prediction)
        return jsonify({'ObeseLevel':str(prediction)})
        
if __name__ == "__main__":
    app.run(debug=True)