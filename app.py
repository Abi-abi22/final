from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Load data and preprocess it here
data = pd.read_csv("gym recommendation.csv")
data.drop(columns=['ID'], inplace=True)

# Label Encoding and Normalization
label_enc = LabelEncoder()
for col in ['Sex', 'Hypertension', 'Diabetes', 'Level', 'Fitness Goal', 'Fitness Type']:
    data[col] = label_enc.fit_transform(data[col])

scaler = StandardScaler()
data[['Age', 'Height', 'Weight', 'BMI']] = scaler.fit_transform(data[['Age', 'Height', 'Weight', 'BMI']])

def get_recommendation(user_input):
    # Convert user input to DataFrame for processing
    user_df = pd.DataFrame([user_input])
    
    # Example logic: Filter recommendations based on fitness goal
    if user_input['Fitness Goal'] == 1:  # Assuming 1 is for Weight Loss
        recommendations = data[data['Fitness Goal'] == 1].sample(n=3).to_dict(orient='records')
    else:  # For Weight Gain or other goals
        recommendations = data[data['Fitness Goal'] == 0].sample(n=3).to_dict(orient='records')
    
    return recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []  # Initialize recommendations as an empty list
    if request.method == 'POST':
        user_input = {
            'Sex': int(request.form['sex']),
            'Age': float(request.form['age']),
            'Height': float(request.form['height']),
            'Weight': float(request.form['weight']),
            'Hypertension': int(request.form['hypertension']),
            'Diabetes': int(request.form['diabetes']),
            'BMI': float(request.form['bmi']),
            'Level': int(request.form['level']),
            'Fitness Goal': int(request.form['fitness_goal']),
            'Fitness Type': int(request.form['fitness_type'])
        }
        recommendations = get_recommendation(user_input)  # Get recommendations based on user input
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
