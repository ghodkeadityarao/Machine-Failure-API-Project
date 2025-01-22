# Manufacturing API for Machine Failure Prediction

## Overview
This project provides an API for predicting machine failure based on a dataset that contains various machine parameters. The API allows you to upload a CSV file containing data, train a model on it, and make predictions on new data. The model predicts machine failure based on various machine parameters like air temperature, process temperature, rotational speed, torque, tool wear.

## Features

- **Upload Endpoint**: Upload a CSV file containing machine data such as machine parameters.
- **Train Endpoint**: Train a machine learning model on the uploaded data to predict machine downtime.
- **Predict Endpoint**: Predict machine failure based on input parameters using the trained model.

## Dataset

The dataset used in this project is focused on predicting machine failure based on various operational parameters. Each row in the dataset represents a set of measurements taken during a machine's operation, along with an indicator (Target) showing whether a failure occurred.

You can download the dataset from here (Also, from the above files).

Link - https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification

## Python Files

1. `app.py`: Contains the Flask code to upload, train and predict based on the ML model.
2. `model.py`: Contains the ML model code which is imported in `app.py` file.
3. `algorithm_test.py`: Conatins the hyperparameter tuning code for the `DecisionTreeClassifier` to find the best parameters.

## Installation

Follow the steps below to set up and run the project locally:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```
2. Install the required Python libraries by running the following command:

   ```bash
   pip install -r requirements.txt
   ```
3. The application will store uploaded files and trained models in specific directories. Create these directories:

   ```bash
   mkdir -p data models
   ```
4. Start the Flask application by running:

   ```bash
   python app.py
   ```
   The application will be accessible at http://127.0.0.1:5000.

## API Endpoints

Here we will be using cURL and Postman to run the commands.

1. Upload Endpoint - `/upload`
   
   Open another terminal and run the following command:
   ```bash
   curl -X POST -F "file=@example_data.csv" http://127.0.0.1:5000/upload
   ```
   For this particular project, you can use the following code:
   ```bash
   curl -X POST -F "file=@data/predictive_maintenance.csv" http://127.0.0.1:5000/upload
   ```
   `Output:`
   ![image](https://github.com/user-attachments/assets/63fe132f-c32c-489a-88bb-d0692fb62e39)

2. Train Endpoint - `/train`
   ```bash
   curl -X POST http://127.0.0.1:5000/train
   ```
   `Output:`
   
   ![image](https://github.com/user-attachments/assets/9dfa1377-6130-40bc-aa4c-0f339e13e116)

4. Predict Endpoint - `/predict`

   Open the `Postman` (if not present, download Postman) and follow the following process:
   - Method: POST
   - URL: `http://127.0.0.1:5000/predict`
   - Body:
       1. Select `raw` and then choose `JSON`.
       2. Provide this (sample) JSON body:
          ```bash
          {
          "Air_Temperature":299.1,
          "Process_Temperature":305.9, 
          "Rotational_Speed":1600,
          "Torque":41.2,
          "Tool_wear":7
          }
          ```
       3. Click on `Send`
     
     `Output:`
     ![image](https://github.com/user-attachments/assets/f52b5ce9-e9eb-46e7-8105-290df1db885e)

    
