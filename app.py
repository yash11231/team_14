import numpy as np
from flask import Flask, request, render_template
import pickle

#Create an app object using the Flask class. 
app = Flask(__name__)

#Load the trained model. (Pickle file)
model = pickle.load(open('models/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/questions')
def questions():
    return render_template('questions.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
    # features = [np.array([4, 5, 3000, 1, 1, 1, 30000, 2, 1])]    
    prediction = model.predict(features)  # f
    result = prediction[0]
    result_integer = int(result)
    print("0", result_integer)

    return render_template('results.html', result=result_integer)

if __name__ == "__main__":
    app.run(debug=False)
