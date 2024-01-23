import logging
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def evaluate_model(model_path, dataset_path):
    logger = logging.getLogger(__name__)
    validation_path = os.path.join(dataset_path, "validation_dataset.csv")

    df = pd.read_csv(validation_path)
    df = df.dropna()
    X_test_prepared = df.iloc[:, 0:-1]
    y_test = df.iloc[:, -1:]

    final_model_path = os.path.join(model_path, "final_model.pkl")
    final_model = joblib.load(open(final_model_path, "rb"))
    score = final_model.score(X_test_prepared, y_test)
    logger.info("Model score: %s", score)
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    logger.info("Mean Sqaure value: %s", final_mse)
    logger.info("Root Mean Sqaure value: %s", final_rmse)

    mlflow.log_metric(key="mse", value=final_mse)
    mlflow.log_metric(key="rmse", value=final_rmse)

    mlflow.log_artifact(dataset_path)
    mlflow.sklearn.log_model(final_model, "Final_model")
    print("Mean Sqaure value: ", final_mse)
    print("Root Mean Sqaure value: ", final_rmse)
