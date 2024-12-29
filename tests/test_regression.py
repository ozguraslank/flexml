import os
import pickle
import unittest
import pandas as pd
from parameterized import parameterized
from sklearn.datasets import load_diabetes
from flexml.regression import Regression
from flexml.logger.logger import get_logger

import warnings
warnings.filterwarnings("ignore")

class TestRegression(unittest.TestCase):
    df = load_diabetes(as_frame=True)['frame']
    logger = get_logger(__name__, "TEST", logging_to_file=False)
    logger.setLevel("DEBUG")
    
    @parameterized.expand(["quick", "wide"])
    def test_regression(self, exp_size: str, df: pd.DataFrame = df):
        try:
            reg_exp = Regression(
                data = df,
                target_col = "target",
                logging_to_file = False
            )
        except Exception as e:
            error_msg = f"An error occured while setting up {exp_size} regression, Error: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

        try:
            reg_exp.start_experiment(
                experiment_size = exp_size,
                test_size = 0.25,
                random_state = 42)
        except Exception as e:
            error_msg = f"An error occured while running {exp_size} regression experiment, Error: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
        top_x_models = reg_exp.get_best_models(top_n_models = 3)
        if len(top_x_models) != 3:
            error_msg = f"An error occured while retriving the best models in {exp_size} regression, expected 3, got {len(top_x_models)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
                
        try:
            reg_exp.show_model_stats()
        except Exception as e:
            error_msg = f"An error occured while showing models stats in {exp_size} regression, Error: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
        try:
            tuning_methods = ["randomized_search", "optuna"]

            for method in tuning_methods:
                reg_exp.tune_model(n_iter=3, n_folds = 3)

                if reg_exp.tuned_model is None:
                    error_msg = f"An error occured while tuning the model with {method} in {exp_size} regression, tuned model is None"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)
                            
                if reg_exp.tuned_model_score is None:
                    error_msg = f"An error occured while calculating the tuned model's score with {method} in {exp_size} regression, tuned model score is None"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)
                    
        except Exception as e:
            error_msg = f"An error occured while tuning the model in {exp_size} regression, Error: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def test_save_model(self):
        try:
            reg_exp = Regression(
                data=self.df,
                target_col="target",
                logging_to_file=False
            )
            reg_exp.start_experiment(
                experiment_size="quick",
                test_size=0.25,
                random_state=42,
            )
            save_path = "test_regression_model.pkl"
            reg_exp.save_model(save_path=save_path)
            self.assertTrue(os.path.exists(save_path))
            with open(save_path, 'rb') as f:
                model = pickle.load(f)
            self.assertIsNotNone(model)
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)