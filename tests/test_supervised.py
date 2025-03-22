import os
import pickle
import unittest
import numpy as np
from parameterized import parameterized
from sklearn.datasets import load_diabetes, load_breast_cancer, load_iris
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
        'BinaryClassification': {
            'data': load_breast_cancer(as_frame=True)['frame'].assign(target=lambda df: df['target'].map({0: 'No', 1: 'Yes'})),
            'target_col': 'target',
            'exp_obj': None
        },
        'MulticlassClassification': {
            'data': load_iris(as_frame=True)['frame'].assign(target=lambda df: df['target'].map({0: 'Iris-Setosa', 1: 'Iris-Versicolor', 2: 'Iris-Virginica'})),
            'target_col': 'target',
            'exp_obj': None
        }
    }
    
    @parameterized.expand(list(test_config.keys()))
    def test_01_supervised(self, objective: str):
        df = self.test_config[objective].get('data')
        target_col = self.test_config[objective].get('target_col')
        exp_size = "wide"
        
        if objective == 'Regression':
            exp_obj = Regression(
                data = df,
                target_col = target_col
            )
        elif objective in ['BinaryClassification', 'MulticlassClassification']:
            exp_obj = Classification(
                data = df,
                target_col = target_col
            )

        exp_obj.start_experiment(
            experiment_size = exp_size
        )
        
        top_x_models = exp_obj.get_best_models(top_n_models = 3)
        self.assertEqual(
            len(top_x_models), 3, 
            f"An error occured while retriving the best models in {exp_size} {objective}, expected 3, got {len(top_x_models)}"
        )
                
        exp_obj.show_model_stats()
        
        tuning_methods = ["randomized_search", "optuna"]
        for method in tuning_methods:
            exp_obj.tune_model(n_iter=3, n_folds = 3)
            self.assertIsNotNone(exp_obj.tuned_model, f"An error occured while tuning the model with {method} in {exp_size} {objective}, tuned model is None")
            self.assertIsNotNone(exp_obj.tuned_model_score, f"An error occured while calculating the tuned model's score with {method} in {exp_size} {objective}, tuned model score is None")            
        
        # Save experiment objects to config
        self.test_config[objective]['exp_obj'] = exp_obj

    def test_02_save_regression_model(self):
        exp_obj = self.test_config['Regression']['exp_obj']

        # Test saving model with full_train=True and model_only=True (only the model object, not a pipeline)
        save_path = "test_regression_model_full_train_model_only.pkl"
        exp_obj.save_model(save_path=save_path, full_train=True, model_only=True)
        self.assertTrue(os.path.exists(save_path))

        # Load the saved model and check if it's the model object (not a pipeline)
        with open(save_path, 'rb') as f:
            saved_model = pickle.load(f)
            self.assertFalse(hasattr(saved_model, 'named_steps'))
        os.remove(save_path) # Clean up saved model

        # Test saving model with full_train=False and model_only=False (should return a pipeline)
        save_path = "test_regression_model_no_full_train_model_only_false.pkl"
        exp_obj.save_model(save_path=save_path, full_train=False, model_only=False)
        self.assertTrue(os.path.exists(save_path))

        # Load the saved model and check if it's a pipeline
        with open(save_path, 'rb') as f:
            saved_model = pickle.load(f)
            self.assertTrue(hasattr(saved_model, 'named_steps'))
        os.remove(save_path) # Clean up saved model
            
    def test_03_save_binary_classification_model(self):
        exp_obj = self.test_config['BinaryClassification']['exp_obj']

        # Test saving model with full_train=True and model_only=True (only the model object, not a pipeline)
        save_path = "test_binary_classification_model_full_train_model_only.pkl"
        exp_obj.save_model(save_path=save_path, full_train=True, model_only=True)
        self.assertTrue(os.path.exists(save_path))

        # Load the saved model and check if it's the model object (not a pipeline)
        with open(save_path, 'rb') as f:
            saved_model = pickle.load(f)
            self.assertFalse(hasattr(saved_model, 'named_steps'))
        os.remove(save_path) # Clean up saved model

        # Test saving model with full_train=False and model_only=False (should return a pipeline)
        save_path = "test_binary_classification_model_no_full_train_model_only_false.pkl"
        exp_obj.save_model(save_path=save_path, full_train=False, model_only=False)
        self.assertTrue(os.path.exists(save_path))

        # Load the saved model and check if it's a pipeline
        with open(save_path, 'rb') as f:
            saved_model = pickle.load(f)
            self.assertTrue(hasattr(saved_model, 'named_steps'))
        os.remove(save_path) # Clean up saved model

    def test_04_save_multiclass_classification_model(self):
        exp_obj = self.test_config['MulticlassClassification']['exp_obj']

        # Test saving model with full_train=True and model_only=True (only the model object, not a pipeline)
        save_path = "test_multiclass_classification_model_full_train_model_only.pkl"
        exp_obj.save_model(save_path=save_path, full_train=True, model_only=True)
        self.assertTrue(os.path.exists(save_path))

        # Load the saved model and check if it's the model object (not a pipeline)
        with open(save_path, 'rb') as f:
            saved_model = pickle.load(f)
            self.assertFalse(hasattr(saved_model, 'named_steps'))
        os.remove(save_path) # Clean up saved model

        # Test saving model with full_train=False and model_only=False (should return a pipeline)
        save_path = "test_multiclass_classification_model_no_full_train_model_only_false.pkl"
        exp_obj.save_model(save_path=save_path, full_train=False, model_only=False)
        self.assertTrue(os.path.exists(save_path))

        # Load the saved model and check if it's a pipeline
        with open(save_path, 'rb') as f:
            saved_model = pickle.load(f)
            self.assertTrue(hasattr(saved_model, 'named_steps'))
        os.remove(save_path) # Clean up saved model

    def test_05_predict_model_regression(self):
        # Test regression predictions
        exp_obj = self.test_config['Regression']['exp_obj']
        test_data = self.test_config['Regression'].get('data').drop(columns=['target'])
        
        predictions = exp_obj.predict(test_data, full_train=False)
        self.assertIsInstance(predictions, np.ndarray)

    def test_06_predict_model_binary_classification(self):
        # Test binary classification predictions
        exp_obj = self.test_config['BinaryClassification']['exp_obj']
        test_data = self.test_config['BinaryClassification'].get('data').drop(columns=['target'])
        
        predictions = exp_obj.predict(test_data, full_train=False)
        predictions_probabilities = exp_obj.predict_proba(test_data, full_train=False)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(predictions_probabilities, np.ndarray)
        self.assertEqual(predictions_probabilities.shape[1], 2)  # Binary classification should have 2 probability columns

    def test_07_predict_model_multiclass(self):
        # Test multiclass classification predictions
        exp_obj = self.test_config['MulticlassClassification']['exp_obj']
        test_data = self.test_config['MulticlassClassification'].get('data').drop(columns=['target'])
        
        predictions = exp_obj.predict(test_data, full_train=False)
        predictions_probabilities = exp_obj.predict_proba(test_data, full_train=False)
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(predictions_probabilities, np.ndarray)
        self.assertEqual(predictions_probabilities.shape[1], 3)  # Iris has 3 classes