🌸 Iris Species Predictor 

This project predicts the species of Iris flowers (Setosa, Versicolor, Virginica) using a Deep Learning Artificial Neural Network (ANN).
It also includes a Flask web application with a clean UI that displays both the prediction and the flower image.

🚀 Features

Trains an ANN model with:

Input features: Sepal length, Sepal width, Petal length, Petal width

Hidden layers with ReLU activation

Output layer with Softmax activation

Optimizer: Adam

Uses SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.

Saves the trained model (.h5) and dataset (.csv).

Flask web app for easy interaction:

Enter flower measurements.

Get predicted species.

See corresponding flower image.

📂 Project Structure

Iris-Predictor/
│
├── static/  

│   ├── setosa.jpg

│   ├── versicolor.jpg

│   └── virginica.jpg
│
├── templates/
│   └── index.html          
│
├── Iris_extended.csv    

├── Iris_extended_SMOTE.csv 

├── Iris_ANN_SMOTE.h5    

├── app.py           

├── training.ipynb       

└── README.md               

⚙️ Installation

Clone this repository:

git clone https://github.com/your-username/Iris-Predictor.git

cd Iris-Predictor


Install dependencies:

pip install -r requirements.txt

Example requirements.txt:

flask

pandas

numpy

scikit-learn

imbalanced-learn

tensorflow

seaborn

matplotlib


📊 Visualizations

Example plots used in training:

Pairplot of features with sns.pairplot

Distribution of balanced dataset after SMOTE

🖼️ Demo

Input: Sepal length = 5.1, Sepal width = 3.5, Petal length = 1.4, Petal width = 0.2

Output: 🌸 Predicted Species: Setosa

Image: Shows Setosa flower.

🧠 Model Summary

Input: 4 features

Hidden Layer 1: Dense(16, ReLU)

Dropout(0.2)

Hidden Layer 2: Dense(12, ReLU)

Dropout(0.2)

Output Layer: Dense(3, Softmax)

Optimizer: Adam

Loss: Categorical Crossentropy

Metric: Accuracy

🙌 Acknowledgements

Dataset: Iris dataset (extended + SMOTE).

Libraries: TensorFlow/Keras, Flask, scikit-learn, imbalanced-learn, seaborn, matplotlib.
