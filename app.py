from flask import Flask, render_template, request
import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the form
        age = float(request.form['age'])
        intraocular_pressure = float(request.form['intraocular_pressure'])
        family_history = float(request.form['family_history'])

        # Preprocess the input data
        input_data = pd.DataFrame({'age': [age],
                                  'intraocular_pressure': [intraocular_pressure],
                                  'family_history': [family_history]})

        input_data_scaled = scaler.transform(input_data)

        # Make a prediction using the trained model
        prediction = model.predict(input_data_scaled)

        # Convert the prediction to a human-readable result
        if prediction[0] == 1:
            result = "High risk of glaucoma"
        else:
            result = "Low risk of glaucoma"

        return render_template('index.html', prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)
