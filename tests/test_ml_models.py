from parameterized import parameterized
import unittest
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer

from flexml.regression import Regression
from flexml.classification import Classification
from flexml.logger.logger import get_logger
from flexml.config.ml_models import WIDE_REGRESSION_MODELS, WIDE_CLASSIFICATION_MODELS

import warnings
warnings.filterwarnings("ignore")

class TestMLModels(unittest.TestCase):
    logger = get_logger(__name__, "TEST", logging_to_file=False)
    logger.setLevel("DEBUG")

    reg_df = load_diabetes(as_frame=True)['frame']
    classification_df = load_breast_cancer(as_frame=True)['frame']

    reg_exp = Regression(
        data = reg_df,
        target_col = "target",
        experiment_size = "wide",
        test_size = 0.25,
        random_state = 42,
        logging_to_file = False
    )

    classification_exp = Classification(
        data = classification_df,
        target_col = "target",
        experiment_size = "wide",
        test_size = 0.25,
        random_state = 42,
        logging_to_file = False
    )

    @parameterized.expand([(model_pack['name'], model_pack['model'], model_pack['tuning_param_grid']) for model_pack in WIDE_REGRESSION_MODELS])
    def test_regression_ml_models(self, model_name, model, model_tuning_params):
        try:
            model.fit(self.reg_exp.X_train, self.reg_exp.y_train)
        except Exception as e:
            error_msg = f"An error occurred while fitting {model_name} model. Error: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

        try:
            self.reg_exp.tune_model(
                model=model,
                tuning_method="randomized_search",
                tuning_size='wide',
                param_grid=model_tuning_params,
                n_iter=3,
                cv=None,
                n_jobs=-1
            )
        except Exception as e:
            error_msg = f"An error occurred while tuning {model_name} model with the following param_grid {model_tuning_params}. Error: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    @parameterized.expand([(model_pack['name'], model_pack['model'], model_pack['tuning_param_grid']) for model_pack in WIDE_CLASSIFICATION_MODELS])
    def test_classification_ml_models(self, model_name, model, model_tuning_params):
        try:
            model.fit(self.classification_exp.X_train, self.classification_exp.y_train)
        except Exception as e:
            error_msg = f"An error occurred while fitting {model_name} model. Error: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

        try:
            self.classification_exp.tune_model(
                model=model,
                tuning_method="randomized_search",
                tuning_size='wide',
                param_grid=model_tuning_params,
                n_iter=3,
                cv=None,
                n_jobs=-1
            )
        except Exception as e:
            error_msg = f"An error occurred while tuning {model_name} model with the following param_grid {model_tuning_params}. Error: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
