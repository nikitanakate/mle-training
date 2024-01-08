import os
import tarfile

import numpy as np
import pandas as pd
from scipy.stats import randint
from six.moves import urllib
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor
import argparse
import logging
from logging.config import fileConfig
import configparser

if __name__ == "__main__":
    fileConfig('logging_config.ini')

    logger = logging.getLogger(__name__)
    config = configparser.ConfigParser()
    config.read('config.ini')
        
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the output folder")
    parser.add_argument("test_dataset_path", help="Path to the testing dataset")
    args = parser.parse_args()

    # Check if the model_path argument is passed
    if args.model_path:
        model_path = args.model_path
        logger.debug('Model path from argument: %s', model_path)
    else:
        # Read the model_path from the config file
        model_path = config.get('Paths', 'model_path')
        logger.debug('Model path from config file: %s', model_path)
        
    # Check if the test_dataset_path argument is passed
    if args.test_dataset_path:
        dataset_path = args.test_dataset_path
        logger.debug('Test dataset path from argument: %s', test_dataset_path)
    else:
        # Read the test_dataset_path from the config file
        dataset_path = config.get('Paths', 'data_path')
        logger.debug('Test dataset path from config file: %s', test_dataset_path)
        
    validation_path = os.path.join(data_path, "validation_dataset.csv")

    df = pd.read_csv(validation_path)
    X_test_prepared = df.iloc[:, 0:-1]
    y_test = df.iloc[:, -1:]
     
    final_model = joblib.load(open(model_path, 'rb')) 
    score = final_model.score(X_test, Y_test)
    logger.debug('Model score: %s', score)
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    logger.debug('Mean Sqaure value: %s', final_mse)
    logger.debug('Root Mean Sqaure value: %s', final_rmse)
