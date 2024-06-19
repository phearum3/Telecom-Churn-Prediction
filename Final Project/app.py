from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import os


app = Flask(__name__)


def train_model():
   # Provide the correct path to your CSV file
    csv_path = os.path.join('c:\\', 'Users', 'Phearum', 'OneDrive', 'Documents', 'Tux Global', 'Data science and Analytics', 'Final Project', 'Data.csv')
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    # Dummy training data
    X_train = df.drop('Churn', axis=1)
    y_train = df['Churn']

    estimators = []
    model1 = DecisionTreeClassifier()
    estimators.append(('decision_tree', model1))
    model2 = RandomForestClassifier()
    estimators.append(('random_forest', model2))
    model3 = GradientBoostingClassifier()
    estimators.append(('gradient_boosting', model3))

    ensemble = VotingClassifier(estimators)
    ensemble.fit(X_train, y_train)
    
    return ensemble

model = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = int(request.form['gender'])
        SeniorCitizen = int(request.form['SeniorCitizen'])
        Partner = int(request.form['Partner'])
        Dependents = int(request.form['Dependents'])
        tenure = int(request.form['tenure'])
        PhoneService = int(request.form['PhoneService'])
        MultipleLines = int(request.form['MultipleLines'])
        InternetService = int(request.form['InternetService'])
        OnlineSecurity = int(request.form['OnlineSecurity'])
        OnlineBackup = int(request.form['OnlineBackup'])
        DeviceProtection = int(request.form['DeviceProtection'])
        TechSupport = int(request.form['TechSupport'])
        StreamingTV = int(request.form['StreamingTV'])
        StreamingMovies = int(request.form['StreamingMovies'])
        Contract = int(request.form['Contract'])
        PaperlessBilling = int(request.form['PaperlessBilling'])
        PaymentMethod = int(request.form['PaymentMethod'])
        MonthlyCharges = float(request.form['MonthlyCharges'])
        TotalCharges = float(request.form['TotalCharges'])
        numAdminTickets = int(request.form['numAdminTickets'])
        numTechTickets = int(request.form['numTechTickets'])

        new_customer = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [SeniorCitizen],
            'Partner': [Partner],
            'Dependents': [Dependents],
            'tenure': [tenure],
            'PhoneService': [PhoneService],
            'MultipleLines': [MultipleLines],
            'InternetService': [InternetService],
            'OnlineSecurity': [OnlineSecurity],
            'OnlineBackup': [OnlineBackup],
            'DeviceProtection': [DeviceProtection],
            'TechSupport': [TechSupport],
            'StreamingTV': [StreamingTV],
            'StreamingMovies': [StreamingMovies],
            'Contract': [Contract],
            'PaperlessBilling': [PaperlessBilling],
            'PaymentMethod': [PaymentMethod],
            'MonthlyCharges': [MonthlyCharges],
            'TotalCharges': [TotalCharges],
            'numAdminTickets': [numAdminTickets],
            'numTechTickets': [numTechTickets]
        })

        prediction = model.predict(new_customer)
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
