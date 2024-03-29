import logging
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor


def train_model(output_folder_path, dataset_path):
    logger = logging.getLogger(__name__)
    train_dataset_path = os.path.join(dataset_path, "training_dataset.csv")
    df = pd.read_csv(train_dataset_path)
    df = df.dropna()
    housing_prepared = df.iloc[:, 0:-1]
    housing_labels = df.iloc[:, -1:]
    logger.info("Read data from training dataset")

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    logger.info("Trained Linear regression model")

    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    logger.debug("Mean Sqaure value for Linear Regression: %s", lin_mse)
    logger.debug("Root Mean Sqaure value for Linear Regression: %s", lin_rmse)

    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    logger.debug("Mean Absolute Error for Linear Regression: %s", lin_mae)

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    logger.debug("Mean Sqaure value for Decision Tree: %s", tree_mse)
    logger.debug("Root Mean Sqaure value for Decision Tree: %s", tree_rmse)

    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4],
        },
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)

    grid_search.best_params_
    mlflow.log_params(grid_search.best_params_)
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

    final_model = grid_search.best_estimator_

    try:
        model_path = os.path.join(output_folder_path, "final_model.pkl")
        # Save the model as a pickle in a file
        joblib.dump(final_model, model_path)
        logger.debug("Model has been saved on this path: %s", model_path)
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
