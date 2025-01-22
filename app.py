import os

import pandas as pd
from flask import Flask, jsonify, request

from model import predict_downtime, train_model

app = Flask(__name__)

UPLOAD_FOLDER = 'data/'
MODEL_FOLDER = 'models/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        return jsonify({"message": "File uploaded successfully.", "columns": list(pd.read_csv(file_path).columns)})


@app.route('/train', methods=['POST'])
def train():
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predictive_maintenance.csv') 
    if not os.path.exists(file_path):
        return jsonify({"error": "No dataset uploaded."}), 400

    metrics = train_model(file_path)
    return jsonify({"message": "Model trained successfully.", "metrics": metrics})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        prediction = predict_downtime(input_data)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    app.run(debug=True)
