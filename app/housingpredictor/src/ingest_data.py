import logging
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HOUSING_PATH = os.path.join("datasets", "housing")
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    # def __init__(self, add_bedrooms_per_room):
    #     self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        total_rooms_index = 3
        population_index = 5
        households_index = 6
        total_bedrooms_index = 4
        rooms_per_household = X[:, total_rooms_index] / X[:, households_index]
        population_per_household = (
            X[:, population_index] / X[:, households_index]
        )
        bedrooms_per_room = X[:, total_bedrooms_index] / X[:, total_rooms_index]
        return np.c_[
            X, rooms_per_household, population_per_household, bedrooms_per_room
        ]

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Get the raw dataset from URL and save it in input path in CSV format

    Parameters
    ----------
    arg1 : string
    URL from where dataset should be downloaded
    arg2: string
    Folder path where data should be stored in CSV format

    Raises
    ------
    Exception if not able to create file

    Notes
    -----
    This function fetches data from given URL and save it in local in CSV format
    """
    try:
        os.makedirs(housing_path, exist_ok=True)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()
    except Exception as e:
        logger.error(f"Failed to fetch housing data: {e}")


def load_housing_data(housing_path=HOUSING_PATH):
    """
    Read the data from CSV file and save it in Pandas dataframe

    Parameters
    ----------
    arg1 : string
    Folder path for given CSV file

    Returns
    -------
    Pandas dataframe

    Raises
    ------
    Exception if not able to read the file

    Notes
    -----
    This function reads data from CSV file and creates Pandas dataframe
    """
    try:
        csv_path = os.path.join(housing_path, "housing.csv")
        return pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to load housing data: {e}")
        return pd.DataFrame()


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


def ingest_data(output_folder_path):
    logger = logging.getLogger(__name__)
    fetch_housing_data()
    logger.info("Fetched data from URL")

    housing = load_housing_data()
    logger.info("Ingested raw data in csv file")

    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    train_set, test_set = train_test_split(
        housing, test_size=0.2, random_state=42
    )
    logger.info("Splitted data into test and train dataset")

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    corr_matrix = housing.corr(numeric_only=True)
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = (
        housing["total_rooms"] / housing["households"]
    )
    housing["bedrooms_per_room"] = (
        housing["total_bedrooms"] / housing["total_rooms"]
    )
    housing["population_per_household"] = (
        housing["population"] / housing["households"]
    )

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_num = housing.drop("ocean_proximity", axis=1)

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
            ("std_scaler", StandardScaler()),
        ]
    )
    # total_rooms_index = housing_num.columns.get_loc("total_rooms")
    # population_index = housing_num.columns.get_loc("population")
    # households_index = housing_num.columns.get_loc("households")
    # total_bedrooms_index = housing_num.columns.get_loc("total_bedrooms")

    # print("total_rooms_index: ", total_rooms_index)
    # print("population_index: ", population_index)
    # print("households_index: ", households_index)
    # print("total_bedrooms_index: ", total_bedrooms_index)

    housing_num_tr = num_pipeline.fit_transform(housing_num)
    housing_tr = pd.DataFrame(
        housing_num_tr,
        columns=list(housing_num.columns)
        + [
            "rooms_per_household",
            "population_per_household",
            "bedrooms_per_room",
        ],
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(
        pd.get_dummies(housing_cat, drop_first=True)
    )

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)

    X_test_prepared_num = num_pipeline.fit_transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared_num,
        columns=list(housing_num.columns)
        + [
            "rooms_per_household",
            "population_per_household",
            "bedrooms_per_room",
        ],
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(
        pd.get_dummies(X_test_cat, drop_first=True)
    )

    try:
        training_dataset = pd.concat([housing_prepared, housing_labels], axis=1)
        validation_dataset = pd.concat([X_test_prepared, y_test], axis=1)

        training_path = os.path.join(output_folder_path, "training_dataset.csv")
        validation_path = os.path.join(
            output_folder_path, "validation_dataset.csv"
        )

        training_dataset.to_csv(training_path)
        validation_dataset.to_csv(validation_path)
        logger.debug("Training dataset saved on this path: %s", training_path)
        logger.debug(
            "Validation dataset saved on this path: %s", validation_path
        )

    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
