from flask import Flask,render_template,jsonify,request,redirect
import pickle
import numpy as np

app = Flask(__name__)
model  = pickle.load(open('model.pkl','rb'))

@app.route('/') 
def home():
    return render_template("home.html")

@app.route('/types')
def types():
    return render_template("types.html")
@app.route('/healthtips')
def healthtips():
    return render_template("healthtips.html")

@app.route('/checkquality')
def checkquality():
    return render_template("checkquality.html")

@app.route('/predict',methods=['POST'])
def predict():

    '''
    For rendering results on HTML GUI
    '''
    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    if prediction == 0:
        output = "edible"
    else:
        output = "poisonous"

    return render_template('result.html', prediction_text='1 is poisonous and 0 is edible.Your mushroom is: {}'.format(output))




if __name__ =='__main__':
    app.run(debug=True)    