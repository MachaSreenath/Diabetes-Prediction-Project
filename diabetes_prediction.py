from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

# Load the dataset
diabetes_dataset = pd.read_csv('C:\\Users\\sree\\Downloads\\Diabetes_Prediction-main\\Diabetes_Prediction-main\\diabetes.csv')

# Divide the dataset into features and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = X_scaled

# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, random_state=2)

# Train the SVM model
svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(X_train, Y_train)

# Train the AdaBoost model
adaboost_classifier = AdaBoostClassifier(n_estimators=50, random_state=2)
adaboost_classifier.fit(X_train, Y_train)

# Initialize Flask app
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for receiving the form data and making a prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    Pregnancies = float(request.form['Pregnancies'])
    Glucose = float(request.form['Glucose'])
    BloodPressure = float(request.form['BloodPressure'])
    SkinThickness = float(request.form['SkinThickness'])
    Insulin = float(request.form['Insulin'])
    BMI = float(request.form['BMI'])
    DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    Age = float(request.form['Age'])
    
    # Create a numpy array with the input values
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    # Standardize the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make predictions using both classifiers
    svm_prediction = svm_classifier.predict(input_data_scaled)
    adaboost_prediction = adaboost_classifier.predict(input_data_scaled)
    
    # Determine the final prediction
    final_prediction = 1 if (svm_prediction + adaboost_prediction) >= 1 else 0
    
    # Determine the prediction message and medication information
    if final_prediction == 1:
        result = 'The person is diabetic.'
        medication = 'Medicines: Metformin, Insulin. Measures: Monitor blood sugar levels regularly, maintain a healthy diet, exercise regularly.'
    else:
        result = 'The person is not diabetic.'
        medication = ''
    
    # Render the result page with the prediction message and medication information
    return render_template('result.html', result=result, medication=medication)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
