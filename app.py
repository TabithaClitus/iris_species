from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder

app = Flask(__name__)

# Load trained ANN model
model = load_model("Iris_ANN_SMOTE.h5")

# Load dataset to fit scaler and encoder
df = pd.read_csv("Iris_extended_SMOTE.csv")
X_columns = df.drop('Species', axis=1).columns

scaler = StandardScaler().fit(df[X_columns])
encoder = OneHotEncoder(sparse_output=False).fit(df[['Species']])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data['sepal_length'], data['sepal_width'],
                          data['petal_length'], data['petal_width']]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    class_index = np.argmax(prediction)
    species_labels = encoder.categories_[0]  # ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    class_name = species_labels[class_index].replace("Iris-", "").capitalize()

    
    return jsonify({'species': class_name})

if __name__ == "__main__":
    app.run(debug=True)
