import unittest
from parameterized import parameterized
from sklearn.datasets import load_breast_cancer, load_diabetes
from flexml.logger.logger import get_logger
from flexml import Regression, Classification

import warnings
warnings.filterwarnings("ignore")

class TestCrossValidation(unittest.TestCase):
    logger = get_logger(__name__, "TEST", logging_to_file=False)

    # Datasets for testing
    regression_data = load_diabetes(as_frame=True)['frame']
    breast_data = load_breast_cancer(as_frame=True)
    classification_data = breast_data['frame']
    classification_data["target"] = breast_data['target']
    classification_data["group"] = classification_data.index % 5  # Add synthetic group column
    regression_data["group"] = regression_data.index % 5  # Add synthetic group column

    @parameterized.expand([
        ("Classification", "kfold", {"n_splits": 5, "test_size": None, "groups_col": None}),
        ("Classification", "stratified_kfold", {"n_splits": 5, "test_size": None, "groups_col": None}),
        ("Classification", "shuffle_split", {"n_splits": 5, "test_size": 0.2, "groups_col": None}),
        ("Classification", "stratified_shuffle_split", {"n_splits": 5, "test_size": 0.2, "groups_col": None}),
        ("Classification", "group_kfold", {"n_splits": 5, "test_size": None, "groups_col": "group"}),
        ("Classification", "group_shuffle_split", {"n_splits": 5, "test_size": 0.2, "groups_col": "group"}),
        ("Classification", "holdout", {"n_splits": None, "test_size": 0.2, "groups_col": None}),
        ("Regression", "kfold", {"n_splits": 5, "test_size": None, "groups_col": None}),
        ("Regression", "shuffle_split", {"n_splits": 5, "test_size": 0.2, "groups_col": None}),
        ("Regression", "group_kfold", {"n_splits": 5, "test_size": None, "groups_col": "group"}),
        ("Regression", "group_shuffle_split", {"n_splits": 5, "test_size": 0.2, "groups_col": "group"}),
        ("Regression", "holdout", {"n_splits": None, "test_size": 0.2, "groups_col": None}),
    ])
    def test_cross_validation(
        self,
        ml_task_type: str,
        cv_method: str,
        params: dict
    ):
        target_col = "target"

        if ml_task_type == "Classification":
            df = self.classification_data.copy()

            # Skip Stratified methods if classes are not sufficiently populated
            if "Stratified" in cv_method and (df["target"].value_counts() < 2).any():
                self.fail(f"{cv_method} couldn't be executed due to insufficient class instances, please take a look to data used for the test")

            experiment_object = Classification(df, target_col)
            
        elif ml_task_type == "Regression":
            if "Stratified" in cv_method:  # Stratified is for Classification only
                self.fail("Stratified methods are for Classification only. You've passed {cv_method} for Regression")

            df = self.regression_data.copy()
            experiment_object = Regression(df, target_col)
        else:
            error_msg = f"Invalid model type specified. Expected 'Classification' or 'Regression', got '{ml_task_type}'"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        experiment_object.start_experiment(
            experiment_size="wide",
            cv_method=cv_method,
            n_folds=params.get("n_splits"),
            test_size=params.get("test_size"),
            groups_col=params.get("groups_col")
            )