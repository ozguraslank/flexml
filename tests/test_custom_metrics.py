import unittest
import numpy as np
from sklearn.datasets import load_diabetes, load_iris, load_breast_cancer
from flexml import Regression, Classification
from flexml.logger import get_logger
import warnings
warnings.filterwarnings("ignore")


class TestCustomMetrics(unittest.TestCase):
    """Test suite for custom scoring function feature"""
    
    logger = get_logger(__name__, "TEST")
    logger.setLevel("DEBUG")
    
    # =========================================================================
    # Custom Metric Functions
    # =========================================================================
    
    @staticmethod
    def custom_mse_doubled(y_true, y_pred):
        """Custom MSE metric multiplied by 2 (for regression)"""
        mse = np.mean((y_true - y_pred) ** 2)
        return mse * 2
    
    @staticmethod
    def custom_accuracy_halved(y_true, y_pred_labels):
        """Custom accuracy metric divided by 2 (for classification with labels)"""
        accuracy = np.mean(y_true == y_pred_labels)
        return accuracy / 2
    
    @staticmethod
    def custom_roc_auc_doubled(y_true, y_proba):
        """Custom ROC-AUC metric multiplied by 2 (for classification with probabilities)
        
        Note: For binary classification, y_proba will be 1D (only positive class probabilities)
        For multiclass classification, y_proba will be 2D (all class probabilities)
        """
        from sklearn.metrics import roc_auc_score
        
        # Handle multiclass case (2D array with >2 classes)
        if len(y_proba.shape) > 1 and y_proba.shape[1] > 2:
            # Multiclass: use OVR strategy
            roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        else:
            # Binary classification: y_proba is already 1D with positive class probabilities
            # OR it's a 2D array with shape (n_samples, 1)
            if len(y_proba.shape) > 1:
                y_proba = y_proba.ravel()  # Flatten if needed
            roc_auc = roc_auc_score(y_true, y_proba)
        
        return roc_auc * 2
    
    # =========================================================================
    # Test 1: Regression with Custom Metric
    # =========================================================================
    
    def test_01_regression_custom_metric(self):
        """Test regression task with custom metric (MSE * 2)"""
        self.logger.info("=" * 80)
        self.logger.info("TEST 1: Regression with Custom Metric (MSE * 2)")
        self.logger.info("=" * 80)
        
        # Load diabetes dataset
        df = load_diabetes(as_frame=True)['frame']
        
        # Create regression model
        model = Regression(
            data=df,
            target_col='target',
            random_state=42
        )
        
        # Run experiment with custom metric
        model.start_experiment(
            experiment_size='quick',
            eval_metric=self.custom_mse_doubled,
            custom_metric_name='Custom MSE x2',
            custom_metric_direction='minimize',
            custom_metric_needs_proba=False,
            cv_method='kfold',
            n_folds=3,
            n_jobs=1
        )
        
        # Verify custom metric was used
        self.assertIsNotNone(model._model_stats_df)
        self.assertIn('Custom MSE x2', model._model_stats_df.columns)
        
        # Verify metric values are reasonable (should be roughly 2x normal MSE)
        # Normal MSE for this dataset is around 3000-6000, so doubled should be 6000-12000
        custom_metric_values = model._model_stats_df['Custom MSE x2'].values
        self.assertTrue(all(val > 0 for val in custom_metric_values), 
                       "Custom metric values should be positive")
        self.assertTrue(all(val > 1000 for val in custom_metric_values),
                       "Custom MSE x2 values should be reasonably large")
        
        # Verify model selection works
        best_model = model.get_best_models()
        self.assertIsNotNone(best_model)
        
        self.logger.info("✓ Regression with custom metric test PASSED")
    
    # =========================================================================
    # Test 2: Binary Classification with Custom Metric (Labels)
    # =========================================================================
    
    def test_02_binary_classification_custom_metric_labels(self):
        """Test binary classification with custom metric for labels (Accuracy / 2)"""
        self.logger.info("=" * 80)
        self.logger.info("TEST 2: Binary Classification with Custom Metric (Accuracy / 2)")
        self.logger.info("=" * 80)
        
        # Load breast cancer dataset
        df = load_breast_cancer(as_frame=True)['frame']
        
        # Create classification model
        model = Classification(
            data=df,
            target_col='target',
            random_state=42
        )
        
        # Run experiment with custom metric that needs labels
        model.start_experiment(
            experiment_size='quick',
            eval_metric=self.custom_accuracy_halved,
            custom_metric_name='Custom Accuracy / 2',
            custom_metric_needs_proba=False,  # Pass labels, not probabilities
            custom_metric_direction='maximize',
            cv_method='kfold',
            n_folds=3,
            n_jobs=1
        )
        
        # Verify custom metric was used
        self.assertIsNotNone(model._model_stats_df)
        self.assertIn('Custom Accuracy / 2', model._model_stats_df.columns)
        
        # Verify metric values are in expected range (0 to 0.5 since accuracy/2)
        custom_metric_values = model._model_stats_df['Custom Accuracy / 2'].values
        self.assertTrue(all(0 <= val <= 0.5 for val in custom_metric_values),
                       "Custom Accuracy / 2 should be between 0 and 0.5")
        self.assertTrue(all(val > 0.2 for val in custom_metric_values),
                       "Custom Accuracy / 2 should be reasonably high (> 0.2)")
        
        # Verify model selection works
        best_model = model.get_best_models()
        self.assertIsNotNone(best_model)
        
        # Verify that the custom metric parameters were stored
        self.assertTrue(model._is_custom_metric)
        self.assertEqual(model.eval_metric.name, 'Custom Accuracy / 2')
        self.assertEqual(model.eval_metric.needs_proba, False)
        self.assertEqual(model.eval_metric.direction, 'maximize')
        
        self.logger.info("✓ Binary classification with custom metric (labels) test PASSED")
    
    # =========================================================================
    # Test 3: Binary Classification with Custom Metric (Probabilities)
    # =========================================================================
    
    def test_03_binary_classification_custom_metric_proba(self):
        """Test binary classification with custom metric for probabilities (ROC-AUC * 2)"""
        self.logger.info("=" * 80)
        self.logger.info("TEST 3: Binary Classification with Custom Metric (ROC-AUC * 2)")
        self.logger.info("=" * 80)
        
        # Load breast cancer dataset
        df = load_breast_cancer(as_frame=True)['frame']
        
        # Create classification model
        model = Classification(
            data=df,
            target_col='target',
            random_state=42
        )
        
        # Run experiment with custom metric that needs probabilities
        model.start_experiment(
            experiment_size='quick',
            eval_metric=self.custom_roc_auc_doubled,
            custom_metric_name='Custom ROC-AUC x2',
            custom_metric_needs_proba=True,  # Pass probabilities
            custom_metric_direction='maximize',
            cv_method='kfold',
            n_folds=3,
            n_jobs=1
        )
        
        # Verify custom metric was used
        self.assertIsNotNone(model._model_stats_df)
        self.assertIn('Custom ROC-AUC x2', model._model_stats_df.columns)
        
        # Verify metric values are in expected range (0 to 2 since ROC-AUC*2)
        custom_metric_values = model._model_stats_df['Custom ROC-AUC x2'].values
        self.assertTrue(all(0 <= val <= 2 for val in custom_metric_values),
                       "Custom ROC-AUC x2 should be between 0 and 2")
        self.assertTrue(all(val > 1.0 for val in custom_metric_values),
                       "Custom ROC-AUC x2 should be reasonably high (> 1.0)")
        
        # Verify model selection works
        best_model = model.get_best_models()
        self.assertIsNotNone(best_model)
        
        # Verify that the custom metric parameters were stored
        self.assertTrue(model._is_custom_metric)
        self.assertEqual(model.eval_metric.name, 'Custom ROC-AUC x2')
        self.assertEqual(model.eval_metric.needs_proba, True)
        self.assertEqual(model.eval_metric.direction, 'maximize')
        
        self.logger.info("✓ Binary classification with custom metric (probabilities) test PASSED")
    
    # =========================================================================
    # Test 4: Multiclass Classification with Custom Metric (Probabilities)
    # =========================================================================
    
    def test_04_multiclass_classification_custom_metric_proba(self):
        """Test multiclass classification with custom metric for probabilities (ROC-AUC * 2)"""
        self.logger.info("=" * 80)
        self.logger.info("TEST 4: Multiclass Classification with Custom Metric (ROC-AUC * 2)")
        self.logger.info("=" * 80)
        
        # Load iris dataset
        df = load_iris(as_frame=True)['frame']
        
        # Create classification model
        model = Classification(
            data=df,
            target_col='target',
            random_state=42
        )
        
        # Run experiment with custom metric that needs probabilities
        model.start_experiment(
            experiment_size='quick',
            eval_metric=self.custom_roc_auc_doubled,
            custom_metric_name='Custom ROC-AUC x2',
            custom_metric_needs_proba=True,  # Pass probabilities
            custom_metric_direction='maximize',
            cv_method='kfold',
            n_folds=3,
            n_jobs=1
        )
        
        # Verify custom metric was used
        self.assertIsNotNone(model._model_stats_df)
        self.assertIn('Custom ROC-AUC x2', model._model_stats_df.columns)
        
        # Verify metric values are in expected range
        custom_metric_values = model._model_stats_df['Custom ROC-AUC x2'].values
        self.assertTrue(all(0 <= val <= 2 for val in custom_metric_values),
                       "Custom ROC-AUC x2 should be between 0 and 2")
        self.assertTrue(all(val > 1.0 for val in custom_metric_values),
                       "Custom ROC-AUC x2 should be reasonably high (> 1.0)")
        
        # Verify model selection works
        best_model = model.get_best_models()
        self.assertIsNotNone(best_model)
        
        self.logger.info("✓ Multiclass classification with custom metric (probabilities) test PASSED")
    
    # =========================================================================
    # Test 5: Tuning with Custom Metric (Regression)
    # =========================================================================
    
    def test_05_tuning_with_custom_metric_regression(self):
        """Test model tuning with custom metric in regression"""
        self.logger.info("=" * 80)
        self.logger.info("TEST 5: Tuning with Custom Metric (Regression)")
        self.logger.info("=" * 80)
        
        # Load diabetes dataset
        df = load_diabetes(as_frame=True)['frame']
        
        # Create regression model
        model = Regression(
            data=df,
            target_col='target',
            random_state=42
        )
        
        # Run experiment with custom metric
        model.start_experiment(
            experiment_size='quick',
            eval_metric=self.custom_mse_doubled,
            custom_metric_name='Custom MSE x2',
            custom_metric_direction='minimize',
            custom_metric_needs_proba=False,
            cv_method='kfold',
            n_folds=3,
            n_jobs=1
        )
        
        # Tune the best model with custom metric (should inherit parameters)
        model.tune_model(
            tuning_method='randomized_search',
            n_iter=2,  # Small number for quick test
            n_jobs=-1,
            verbose=0
        )
        
        # Verify tuning completed and custom metric was used
        # Check that the leaderboard has a tuned model
        self.assertIsNotNone(model._model_stats_df)
        tuned_models = model._model_stats_df[model._model_stats_df['Model Name'].str.contains('randomized_search', case=False)]
        self.assertTrue(len(tuned_models) > 0, "Tuned model should be added to leaderboard")
        
        self.logger.info("✓ Tuning with custom metric (regression) test PASSED")
    
    # =========================================================================
    # Test 6: Tuning with Custom Metric (Classification)
    # =========================================================================
    
    def test_06_tuning_with_custom_metric_classification(self):
        """Test model tuning with custom metric in classification"""
        self.logger.info("=" * 80)
        self.logger.info("TEST 6: Tuning with Custom Metric (Classification)")
        self.logger.info("=" * 80)
        
        # Load breast cancer dataset
        df = load_breast_cancer(as_frame=True)['frame']
        
        # Create classification model
        model = Classification(
            data=df,
            target_col='target',
            random_state=42
        )
        
        # Run experiment with custom metric
        model.start_experiment(
            experiment_size='quick',
            eval_metric=self.custom_roc_auc_doubled,
            custom_metric_name='Custom ROC-AUC x2',
            custom_metric_needs_proba=True,
            custom_metric_direction='maximize',
            cv_method='kfold',
            n_folds=3,
            n_jobs=1
        )
        
        # Tune the best model with custom metric
        model.tune_model(
            tuning_method='randomized_search',
            n_iter=2,  # Small number for quick test
            n_jobs=-1,
            verbose=0
        )
        
        # Verify tuning completed
        # Check that the leaderboard has a tuned model
        self.assertIsNotNone(model._model_stats_df)
        tuned_models = model._model_stats_df[model._model_stats_df['Model Name'].str.contains('randomized_search', case=False)]
        self.assertTrue(len(tuned_models) > 0, "Tuned model should be added to leaderboard")
        
        self.logger.info("✓ Tuning with custom metric (classification) test PASSED")
    
    # =========================================================================
    # Test 7: Optuna Tuning with Custom Metric (Regression)
    # =========================================================================
    
    def test_07_optuna_tuning_regression(self):
        """Test Optuna tuning with custom metric in regression"""
        self.logger.info("=" * 80)
        self.logger.info("TEST 7: Optuna Tuning with Custom Metric (Regression)")
        self.logger.info("=" * 80)
        
        # Load diabetes dataset
        df = load_diabetes(as_frame=True)['frame']
        
        # Create regression model
        model = Regression(
            data=df,
            target_col='target',
            random_state=42
        )
        
        # Run experiment with custom metric
        model.start_experiment(
            experiment_size='quick',
            eval_metric=self.custom_mse_doubled,
            custom_metric_name='Custom MSE x2',
            custom_metric_direction='minimize',
            custom_metric_needs_proba=False,
            cv_method='kfold',
            n_folds=3,
            n_jobs=-1
        )
        
        # Tune with Optuna (should inherit parameters)
        model.tune_model(
            tuning_method='optuna',
            n_iter=2,  # Small number for quick test
            n_jobs=-1,
            verbose=0
        )
        
        # Verify tuning completed
        self.assertIsNotNone(model._model_stats_df)
        tuned_models = model._model_stats_df[model._model_stats_df['Model Name'].str.contains('optuna', case=False)]
        self.assertTrue(len(tuned_models) > 0, "Optuna tuned model should be added to leaderboard")
        
        # Verify the custom metric was used in tuning
        tuned_model_row = tuned_models.iloc[0]
        self.assertIn('Custom MSE x2', tuned_model_row.index)
        self.assertTrue(tuned_model_row['Custom MSE x2'] > 0, "Custom metric should have valid value")
        
        self.logger.info("✓ Optuna tuning with custom metric (regression) test PASSED")
    
    # =========================================================================
    # Test 8: Optuna Tuning with Custom Metric (Classification)
    # =========================================================================
    
    def test_08_optuna_tuning_classification(self):
        """Test Optuna tuning with custom metric in classification"""
        self.logger.info("=" * 80)
        self.logger.info("TEST 8: Optuna Tuning with Custom Metric (Classification)")
        self.logger.info("=" * 80)
        
        # Load breast cancer dataset
        df = load_breast_cancer(as_frame=True)['frame']
        
        # Create classification model
        model = Classification(
            data=df,
            target_col='target',
            random_state=42
        )
        
        # Run experiment with custom metric that needs probabilities
        model.start_experiment(
            experiment_size='quick',
            eval_metric=self.custom_roc_auc_doubled,
            custom_metric_name='Custom ROC-AUC x2',
            custom_metric_direction='maximize',
            custom_metric_needs_proba=True,
            cv_method='kfold',
            n_folds=3,
            n_jobs=-1
        )
        
        # Tune with Optuna
        model.tune_model(
            tuning_method='optuna',
            n_iter=2,  # Small number for quick test
            n_jobs=-1,
            verbose=0
        )
        
        # Verify tuning completed
        self.assertIsNotNone(model._model_stats_df)
        tuned_models = model._model_stats_df[model._model_stats_df['Model Name'].str.contains('optuna', case=False)]
        self.assertTrue(len(tuned_models) > 0, "Optuna tuned model should be added to leaderboard")
        
        # Verify the custom metric was used in tuning
        tuned_model_row = tuned_models.iloc[0]
        self.assertIn('Custom ROC-AUC x2', tuned_model_row.index)
        self.assertTrue(0 <= tuned_model_row['Custom ROC-AUC x2'] <= 2, 
                       "Custom ROC-AUC x2 should be between 0 and 2")
        
        self.logger.info("✓ Optuna tuning with custom metric (classification) test PASSED")
    
    # =========================================================================
    # Test 9: Invalid Custom Metric Function
    # =========================================================================
    
    def test_09_invalid_custom_metric_function(self):
        """Test that invalid custom metric functions are rejected"""
        self.logger.info("=" * 80)
        self.logger.info("TEST 9: Invalid Custom Metric Function")
        self.logger.info("=" * 80)
        
        # Load dataset
        df = load_diabetes(as_frame=True)['frame']
        
        # Create regression model
        model = Regression(
            data=df,
            target_col='target',
            random_state=42
        )
        
        # Define invalid function (wrong number of parameters)
        def invalid_metric(y_true, y_pred, extra_param):
            return np.mean((y_true - y_pred) ** 2)
        
        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            model.start_experiment(
                experiment_size='quick',
                eval_metric=invalid_metric,
                custom_metric_name='Invalid Metric',
                custom_metric_direction='maximize',
                custom_metric_needs_proba=False,
                cv_method='kfold',
                n_folds=3
            )
        
        self.assertIn("exactly 2 parameters", str(context.exception))
        
        self.logger.info("✓ Invalid custom metric function test PASSED")


if __name__ == '__main__':
    unittest.main()

