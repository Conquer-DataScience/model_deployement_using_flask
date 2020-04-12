import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print(request.form.values())
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    results = model.predict(final_features)
    
    return render_template('index.html',
            prediction_text='Predicted Iris Species: {}'.format(results))

if __name__=="__main__":
    app.run(debug=True)
    

    