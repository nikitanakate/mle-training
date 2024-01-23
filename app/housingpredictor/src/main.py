import argparse
import configparser
import logging
import os
from logging.config import fileConfig

import mlflow
import mlflow.sklearn
from ingest_data import ingest_data
from score import evaluate_model
from train import train_model


def mlflow_run():
    experiment_id = mlflow.set_experiment("House-price-prediction")
    experiment_id = experiment_id.experiment_id
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", help="Path to the dataset folder")
    parser.add_argument("model_path", help="Path to the model folder")
    args = parser.parse_args()

    print("====================================================")
    print(args.dataset_path)
    # Check if the dataset_path argument is passed
    if args.dataset_path:
        dataset_path = args.dataset_path
        logger.debug("Dataset path from argument: %s", dataset_path)
    else:
        # Read the dataset_path from the config file
        dataset_path = config.get("Paths", "data_path")
        logger.debug("Dataset path from config file: %s", dataset_path)

    # Check if the model_path argument is passed
    if args.model_path:
        model_path = args.model_path
        logger.debug("Model path from argument: %s", model_path)
    else:
        # Read the model_path from the config file
        model_path = config.get("Paths", "model_path")
        logger.debug("Model path from config file: %s", model_path)

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    with mlflow.start_run(
        run_name="PARENT_RUN", experiment_id=experiment_id, description="parent"
    ) as parent_run:
        with mlflow.start_run(
            run_name="DATA_LOAD",
            experiment_id=experiment_id,
            description="data_load",
            nested=True,
        ) as data_load_run:
            ingest_data(dataset_path)

        with mlflow.start_run(
            run_name="TRAIN_MODEL",
            experiment_id=experiment_id,
            description="train_model",
            nested=True,
        ) as train_model_run:
            train_model(model_path, dataset_path)

        with mlflow.start_run(
            run_name="EVALUATE_MODEL",
            experiment_id=experiment_id,
            description="evaluate_model",
            nested=True,
        ) as evaluate_model_run:
            evaluate_model(model_path, dataset_path)

        print("MLflow run completed.")


if __name__ == "__main__":
    fileConfig("logging_config.ini")
    logger = logging.getLogger(__name__)
    config = configparser.ConfigParser()
    config.read("config.ini")
    mlflow_run()
