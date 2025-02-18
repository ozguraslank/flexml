import os
import pickle
import unittest
import numpy as np
from parameterized import parameterized
from sklearn.datasets import load_diabetes, load_breast_cancer
from flexml import Regression, Classification
from flexml.logger import get_logger

import warnings
warnings.filterwarnings("ignore")


class TestRegression(unittest.TestCase):
    logger = get_logger(__name__, "TEST")
    logger.setLevel("DEBUG")

    test_config = {
        'Regression': {
            'data': load_diabetes(as_frame=True)['frame'],
            'target_col': 'target',
            'exp_obj': None
        },

        'Classification': {
            'data': load_breast_cancer(as_frame=True)['frame'],
            'target_col': 'target',
            'exp_obj': None
        }
    }
    
    @parameterized.expand(list(test_config.keys()))
    def test_01_supervised(self, objective: str):
        df = self.test_config[objective].get('data')
        target_col = self.test_config[objective].get('target_col')
        exp_size = "wide"
        
        try:
            if objective == 'Regression':
                exp_obj = Regression(
                    data = df,
                    target_col = target_col
                )

            else:
                exp_obj = Classification(
                    data = df,
                    target_col = target_col
                )

        except Exception as e:
            error_msg = f"An error occured while setting up {exp_size} {objective} experiment, Error: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)

        try:
            exp_obj.start_experiment(
                experiment_size = exp_size
            )
            
        except Exception as e:
            error_msg = f"An error occured while running {exp_size} {objective} experiment, Error: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
        top_x_models = exp_obj.get_best_models(top_n_models = 3)
        if len(top_x_models) != 3:
            error_msg = f"An error occured while retriving the best models in {exp_size} {objective}, expected 3, got {len(top_x_models)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
                
        try:
            exp_obj.show_model_stats()
        except Exception as e:
            error_msg = f"An error occured while showing models stats in {exp_size} {objective}, Error: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
        try:
            tuning_methods = ["randomized_search", "optuna"]

            for method in tuning_methods:
                exp_obj.tune_model(n_iter=3, n_folds = 3)

                if exp_obj.tuned_model is None:
                    error_msg = f"An error occured while tuning the model with {method} in {exp_size} {objective}, tuned model is None"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)
                            
                if exp_obj.tuned_model_score is None:
                    error_msg = f"An error occured while calculating the tuned model's score with {method} in {exp_size} {objective}, tuned model score is None"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)
                    
        except Exception as e:
            error_msg = f"An error occured while tuning the model in {exp_size} {objective}, Error: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        
        # Save experiment objects to config
        self.test_config[objective]['exp_obj'] = exp_obj

    def test_02_save_model(self):
        try:
            exp_obj = self.test_config['Regression']['exp_obj']
            
            # Thanks to function naming, test_01_supervised will run first and exp_obj will be created --
            # But let's check it in just case and create a new experiment object if it's None
            if exp_obj is None:
                df = self.test_config['Regression'].get('data')
                target_col = self.test_config['Regression'].get('target_col')

                exp_obj = Regression(
                    data = df,
                    target_col = target_col
                ).start_experiment()

            save_path = "test_regression_model.pkl"
            exp_obj.save_model(save_path=save_path)
            self.assertTrue(os.path.exists(save_path))
            with open(save_path, 'rb') as f:
                model = pickle.load(f)
            self.assertIsNotNone(model)

        finally:
            if os.path.exists(save_path):
                os.remove(save_path)
            
    def test_03_save_model(self):
        try:
            exp_obj = self.test_config['Regression']['exp_obj']
            
            # Thanks to function naming, test_01_supervised will run first and exp_obj will be created --
            # But let's check it in just case and create a new experiment object if it's None
            if exp_obj is None:
                df = self.test_config['Regression'].get('data')
                target_col = self.test_config['Regression'].get('target_col')

                exp_obj = Regression(
                    data = df,
                    target_col = target_col
                ).start_experiment()

            # Test saving model with full_train=True and model_only=True (only the model object, not a pipeline)
            save_path = "test_classification_model_full_train_model_only.pkl"
            exp_obj.save_model(save_path=save_path, full_train=True, model_only=True)

            # Check if the model is saved
            self.assertTrue(os.path.exists(save_path))

            # Load the saved model and check if it's the model object (not a pipeline)
            with open(save_path, 'rb') as f:
                saved_model = pickle.load(f)
                self.assertFalse(hasattr(saved_model, 'named_steps'))

            # Clean up saved model
            os.remove(save_path)

            # Test saving model with full_train=False and model_only=False (should return a pipeline)
            save_path = "test_classification_model_no_full_train_model_only_false.pkl"
            exp_obj.save_model(save_path=save_path, full_train=False, model_only=False)

            # Check if the model is saved
            self.assertTrue(os.path.exists(save_path))

            # Load the saved model and check if it's a pipeline
            with open(save_path, 'rb') as f:
                saved_model = pickle.load(f)
                self.assertTrue(hasattr(saved_model, 'named_steps'))

            # Clean up saved model
            os.remove(save_path)
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_04_predict_model(self):
        # Setup for Classification experiment
        exp_obj = self.test_config['Regression']['exp_obj']
        
        # Thanks to function naming, test_01_supervised will run first and exp_obj will be created --
        # But let's check it in just case and create a new experiment object if it's None
        if exp_obj is None:
            df = self.test_config['Regression'].get('data')
            target_col = self.test_config['Regression'].get('target_col')

            exp_obj = Regression(
                data = df,
                target_col = target_col
            ).start_experiment()

        # Make predictions
        predictions = exp_obj.predict(self.test_config['Regression'].get('data').drop(columns=['target']), full_train=False)
        self.assertIsInstance(predictions, np.ndarray)