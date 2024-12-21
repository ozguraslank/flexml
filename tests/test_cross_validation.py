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
        ("Classification", "k-fold", {"n_splits": 5, "test_size": None, "groups_col": None}),
        ("Classification", "StratifiedKfold", {"n_splits": 5, "test_size": None, "groups_col": None}),
        ("Classification", "ShuffleSplit", {"n_splits": 5, "test_size": 0.2, "groups_col": None}),
        ("Classification", "StratifiedShuffleSplit", {"n_splits": 5, "test_size": 0.2, "groups_col": None}),
        ("Classification", "GroupKFold", {"n_splits": 5, "test_size": None, "groups_col": "group"}),
        ("Classification", "GroupShuffleSplit", {"n_splits": 5, "test_size": 0.2, "groups_col": "group"}),
        ("Classification", "hold-out", {"n_splits": None, "test_size": 0.2, "groups_col": None}),
        ("Regression", "k-fold", {"n_splits": 5, "test_size": None, "groups_col": None}),
        ("Regression", "ShuffleSplit", {"n_splits": 5, "test_size": 0.2, "groups_col": None}),
        ("Regression", "GroupKFold", {"n_splits": 5, "test_size": None, "groups_col": "group"}),
        ("Regression", "GroupShuffleSplit", {"n_splits": 5, "test_size": 0.2, "groups_col": "group"}),
        ("Regression", "hold-out", {"n_splits": None, "test_size": 0.2, "groups_col": None}),
    ])
    def test_cross_validation(self, ml_task_type: str, cv_method: str, params: dict):
        if ml_task_type == "Classification":
            df = self.classification_data.copy()
            y_label = "target"

            # Skip Stratified methods if classes are not sufficiently populated
            if "Stratified" in cv_method and (df["target"].value_counts() < 2).any():
                self.fail(f"{cv_method} couldn't be executed due to insufficient class instances, please take a look to data used for the test")

            experiment_object = Classification(df, y_label)
            
        elif ml_task_type == "Regression":
            if "Stratified" in cv_method:  # Stratified is for Classification only
                self.fail("Stratified methods are for Classification only. You've passed {cv_method} for Regression")

            df = self.regression_data.copy()
            y_label = "target"
            experiment_object = Regression(df, y_label)
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