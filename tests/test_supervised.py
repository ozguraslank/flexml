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

    n_folds = 3
    
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
        else: # BinaryClassification or MulticlassClassification
            exp_obj = Classification(
                data = df,
                target_col = target_col
            )

        exp_obj.start_experiment(
            experiment_size = exp_size,
            n_folds = self.n_folds,
            eval_metric = "RMSE" if objective == "Regression" else "Accuracy"
        )
        
        top_x_models = exp_obj.get_best_models(top_n_models = 3)
        self.assertEqual(
            len(top_x_models), 3, 
            f"An error occured while retriving the best models in {exp_size} {objective}, expected 3, got {len(top_x_models)}"
        )
                
        exp_obj.show_model_stats()
        
        tuning_methods = ["grid_search", "randomized_search", "optuna"]
        for method in tuning_methods:
            if method == "grid_search":
                model = "LGBMRegressor" if objective == "Regression" else "LGBMClassifier"
                param_grid = {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5],
                    "learning_rate": [0.5, 0.1]
                }
                exp_obj.tune_model(model=model, tuning_method=method, param_grid=param_grid, n_folds=self.n_folds, n_iter=3)
            else:
                exp_obj.tune_model(tuning_method=method, n_folds=self.n_folds, n_iter=3)
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
        
        predictions = exp_obj.predict(
            test_data=test_data,
            model=exp_obj.get_model_by_name("LGBMRegressor"),
            full_train=True,
        )
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
    
    def test_08_plot_regression_feature_importance(self):
        exp_obj = self.test_config['Regression']['exp_obj']
        exp_obj.plot("CatBoostRegressor", kind="feature_importance")

    def test_09_plot_binary_classification_feature_importance(self):
        exp_obj = self.test_config['BinaryClassification']['exp_obj']
        exp_obj.plot("XGBClassifier", kind="feature_importance")

    def test_10_plot_multiclass_classification_feature_importance(self):
        exp_obj = self.test_config['MulticlassClassification']['exp_obj']
        exp_obj.plot("LogisticRegression", kind="feature_importance")

    def test_11_plot_regression_residuals(self):
        exp_obj = self.test_config['Regression']['exp_obj']
        exp_obj.plot("LinearRegression", kind="residuals")

    def test_12_plot_regression_prediction_error(self):
        exp_obj = self.test_config['Regression']['exp_obj']
        exp_obj.plot("LGBMRegressor", kind="prediction_error")

    def test_13_plot_regression_shap_summary(self):
        exp_obj = self.test_config['Regression']['exp_obj']
        exp_obj.plot("XGBRegressor", kind="shap_summary")

    def test_14_plot_regression_shap_violin(self):
        exp_obj = self.test_config['Regression']['exp_obj']
        exp_obj.plot("RandomForestRegressor", kind="shap_violin")

    def test_15_plot_binary_classification_confusion_matrix(self):
        exp_obj = self.test_config['BinaryClassification']['exp_obj']
        exp_obj.plot("LogisticRegression", kind="confusion_matrix")

    def test_16_plot_binary_classification_roc_curve(self):
        exp_obj = self.test_config['BinaryClassification']['exp_obj']
        exp_obj.plot("RandomForestClassifier", kind="roc_curve")

    def test_17_plot_binary_classification_calibration_uniform(self):
        exp_obj = self.test_config['BinaryClassification']['exp_obj']
        exp_obj.plot("XGBClassifier", kind="calibration_curve", strategy='uniform', n_bins=10)

    def test_18_plot_binary_classification_calibration_quantile(self):
        exp_obj = self.test_config['BinaryClassification']['exp_obj']
        exp_obj.plot("LGBMClassifier", kind="calibration_curve", strategy='quantile', n_bins=8)

    def test_19_plot_binary_classification_shap_summary(self):
        exp_obj = self.test_config['BinaryClassification']['exp_obj']
        exp_obj.plot("CatBoostClassifier", kind="shap_summary")

    def test_20_plot_binary_classification_shap_violin(self):
        exp_obj = self.test_config['BinaryClassification']['exp_obj']
        exp_obj.plot("XGBClassifier", kind="shap_violin")

    def test_21_plot_multiclass_classification_confusion_matrix(self):
        exp_obj = self.test_config['MulticlassClassification']['exp_obj']
        exp_obj.plot("RandomForestClassifier", kind="confusion_matrix")

    def test_22_plot_multiclass_classification_roc_curve(self):
        exp_obj = self.test_config['MulticlassClassification']['exp_obj']
        exp_obj.plot("LogisticRegression", kind="roc_curve")

    def test_23_plot_multiclass_calibration_uniform(self):
        exp_obj = self.test_config['MulticlassClassification']['exp_obj']
        exp_obj.plot("XGBClassifier", kind="calibration_curve", strategy='uniform', n_bins=10)

    def test_24_plot_multiclass_calibration_quantile(self):
        exp_obj = self.test_config['MulticlassClassification']['exp_obj']
        exp_obj.plot("CatBoostClassifier", kind="calibration_curve", strategy='quantile', n_bins=12)

    def test_25_plot_multiclass_classification_shap_summary(self):
        exp_obj = self.test_config['MulticlassClassification']['exp_obj']
        exp_obj.plot("LGBMClassifier", kind="shap_summary")

    def test_26_plot_multiclass_classification_shap_violin(self):
        exp_obj = self.test_config['MulticlassClassification']['exp_obj']
        exp_obj.plot("RandomForestClassifier", kind="shap_violin")