import unittest
import pandas as pd
import numpy as np
from parameterized import parameterized
from flexml.helpers import validate_inputs, eval_metric_checker, random_state_checker
import warnings
warnings.filterwarnings("ignore")


class TestHelpers(unittest.TestCase):
    """
    Test cases for the feature engineering pipeline in the Classification class
    """
    np.random.seed(42)
    n_rows = 100

    df = pd.DataFrame({
        'id': range(1, n_rows + 1),
        'category_default': np.random.choice(['A', 'B', 'C'], n_rows),    
        'value_default': np.random.normal(100, 15, n_rows),
        'status': np.random.choice(['Active', 'Pending', 'Closed'], n_rows),
        'priority': np.random.choice(['High', 'Medium', 'Low'], n_rows),
        'score': np.random.randint(0, 100, n_rows),
        'amount': np.random.uniform(10, 1000, n_rows),
        'target': np.random.choice([0, 1], n_rows)
    })

    # This will create artificial null values within dataframe
    for column in df.columns:
        if column not in ['id', 'target']:
            mask = np.random.random(n_rows) < 0.2
            df.loc[mask, column] = np.nan

    encoding_methods = ['label_encoder', 'onehot_encoder', 'ordinal_encoder']
    imputation_methods = ['mean', 'median', 'mode', 'constant', 'drop']
    normalization_methods = ['standard_scaler', 'minmax_scaler', 'robust_scaler', 'quantile_transformer', 'maxabs_scaler', 'normalize_scaler']


    @parameterized.expand([
        # Basic validation errors
        (
            "target_in_drop_columns", 
            {"drop_columns": ["target"]}, 
            ValueError, 
            "target column 'target' cannot be in the drop_columns list"
        ),

        (
            "no_features_after_drop", 
            {"drop_columns": ["category_default", "value_default", "status", "priority", "score", "amount", "id"]}, 
            ValueError, 
            "After dropping columns, only {'target'} remain"
        ),

        # Imputation method errors
        (
            "invalid_cat_imputation", 
            {"categorical_imputation_method": "invalid"}, 
            ValueError,
            "categorical_imputation_method 'invalid' is not valid"
        ),

        (
            "invalid_num_imputation", 
            {"numerical_imputation_method": "invalid"}, 
            ValueError,
            "numerical_imputation_method 'invalid' is not valid"
        ),

        # Column imputation map errors
        (
            "column_imputation_invalid_column", 
            {"column_imputation_map": {"nonexistent_column": "mean"}}, 
            ValueError,
            "column 'nonexistent_column' in column_imputation_map is not in the data"
        ),

        (
            "column_imputation_invalid_numeric_method", 
            {"column_imputation_map": {"value_default": "invalid"}}, 
            ValueError,
            "numeric imputation method 'invalid' for column 'value_default' is not valid"
        ),

        (
            "column_imputation_invalid_categorical_method", 
            {"column_imputation_map": {"category_default": "invalid"}}, 
            ValueError,
            "categorical imputation method 'invalid' for column 'category_default' is not valid"
        ),

        # Constant type errors
        (
            "invalid_numerical_constant", 
            {"numerical_imputation_constant": "invalid"}, 
            ValueError,
            "numerical_imputation_constant should be a number"
        ),

        (
            "invalid_categorical_constant", 
            {"categorical_imputation_constant": 123}, 
            ValueError,
            "categorical_imputation_constant should be a string"
        ),

        # Encoding method errors
        (
            "invalid_encoding_method", 
            {"encoding_method": "invalid_encoder"}, 
            ValueError,
            "encoding_method 'invalid_encoder' is not valid"
        ),

        (
            "invalid_onehot_limit", 
            {"onehot_limit": -5}, 
            ValueError,
            "onehot_limit should be a positive integer"
        ),

        # Encoding method map errors
        (
            "encoding_map_invalid_column", 
            {"encoding_method_map": {"nonexistent_column": "label_encoder"}}, 
            ValueError,
            "column 'nonexistent_column' in encoding_method_map is not in the data"
        ),

        (
            "encoding_map_dropped_column", 
            {"drop_columns": ["category_default"], "encoding_method_map": {"category_default": "label_encoder"}}, 
            ValueError,
            "column 'category_default' in encoding_method_map is in drop_columns"
        ),

        (
            "encoding_map_invalid_method", 
            {"encoding_method_map": {"category_default": "invalid"}}, 
            ValueError,
            "encoding method 'invalid' for column 'category_default' is not valid"
        ),

        # Ordinal encoding errors
        (
            "missing_ordinal_map", 
            {"encoding_method": "ordinal_encoder"}, 
            ValueError,
            "Ordinal encoding is selected but no ordinal_encode_map is provided"
        ),

        (
            "missing_column_ordinal_map", 
            {"encoding_method": "ordinal_encoder", "ordinal_encode_map": {}}, 
            ValueError,
            "Ordinal encoding is selected for column 'category_default' but no ordinal_encode_map is provided"
        ),

        (
            "mismatched_ordinal_values", 
            {"encoding_method": "ordinal_encoder", 
             "ordinal_encode_map": {
                 "category_default": ["X", "Y", "Z"],
                 "status": ["Active", "Pending", "Closed"],
                 "priority": ["Low", "Medium", "High"]}}, 
            ValueError,
            "Distinct values in column 'category_default' do not match"
        ),

        (
            "extra_columns_ordinal_map", 
            {"encoding_method": "ordinal_encoder", 
             "ordinal_encode_map": {
                 "category_default": ["A", "B", "C"],
                 "status": ["Active", "Pending", "Closed"],
                 "priority": ["Low", "Medium", "High"],
                 "extra_column": ["X", "Y", "Z"]}}, 
            ValueError,
            "Ordinal_encode_map includes extra columns not in the categorical columns"
        ),

        # Normalization errors
        (
            "invalid_normalization", 
            {"normalize": "invalid_scaler"}, 
            ValueError,
            "normalize method 'invalid_scaler' is not valid"
        ),

        # Drop columns validation
        (
            "drop_column_not_in_data", 
            {"drop_columns": ["nonexistent_column"]}, 
            ValueError,
            "column 'nonexistent_column' in drop_columns is not in the data"
        ),

        # Ordinal encoding in method map errors
        (
            "missing_ordinal_map_in_method_map", 
            {"encoding_method_map": {"category_default": "ordinal_encoder"}}, 
            ValueError,
            "Ordinal encoding is selected for column 'category_default' but no ordinal_encode_map is provided"
        ),

        (
            "missing_column_ordinal_map_in_method_map", 
            {"encoding_method_map": {"category_default": "ordinal_encoder"},
             "ordinal_encode_map": {}}, 
            ValueError,
            "Ordinal encoding is selected for column 'category_default' but no ordinal_encode_map is provided"
        ),

        (
            "mismatched_ordinal_values_in_method_map", 
            {"encoding_method_map": {
                 "category_default": "ordinal_encoder",
                 "status": "label_encoder",
                 "priority": "label_encoder"
             },
             "ordinal_encode_map": {
                 "category_default": ["X", "Y", "Z"]
             }}, 
            ValueError,
            "Unique values in 'category_default' do not match with the ones given in ordinal_encode_map"
        ),

        (
            "extra_columns_ordinal_map_in_method_map", 
            {"encoding_method_map": {
                 "category_default": "ordinal_encoder"
             },
             "ordinal_encode_map": {
                 "category_default": ["A", "B", "C"],
                 "extra_column": ["X", "Y", "Z"]
             }}, 
            ValueError,
            "Ordinal_encode_map includes extra columns not specified for ordinal encoding"
        ),
    ])
    def test_validate_inputs_errors(self, test_name, params, expected_error, expected_message):
        """Test validate_inputs exception raising for invalid parameters"""
        with self.assertRaisesRegex(expected_error, expected_message):
            validate_inputs(
                data=self.df,
                target_col='target',
                **params
            )

    @parameterized.expand([
        # Default behavior tests
        (
            "regression_default",
            {"ml_task_type": "Regression", "eval_metric": None},
            "R2",
            None
        ),
        (
            "classification_default",
            {"ml_task_type": "Classification", "eval_metric": None},
            "Accuracy",
            None
        ),

        # Regression metric tests
        (
            "regression_valid_lowercase",
            {"ml_task_type": "Regression", "eval_metric": "mae"},
            "MAE",
            None
        ),
        (
            "regression_valid_uppercase",
            {"ml_task_type": "Regression", "eval_metric": "RMSE"},
            "RMSE",
            None
        ),
        (
            "regression_invalid_metric",
            {"ml_task_type": "Regression", "eval_metric": "invalid"},
            None,
            ValueError
        ),

        # Classification metric tests
        (
            "classification_valid_exact",
            {"ml_task_type": "Classification", "eval_metric": "Accuracy"},
            "Accuracy",
            None
        ),
        (
            "classification_valid_flexible",
            {"ml_task_type": "Classification", "eval_metric": "roc-auc"},
            "ROC-AUC",
            None
        ),
        (
            "classification_valid_no_special",
            {"ml_task_type": "Classification", "eval_metric": "rocauc"},
            "ROC-AUC",
            None
        ),
        (
            "classification_invalid_metric",
            {"ml_task_type": "Classification", "eval_metric": "invalid"},
            None,
            ValueError
        ),

        # Custom metrics list tests
        (
            "custom_metrics_valid_classification",
            {
                "ml_task_type": "Classification",
                "eval_metric": "F1 Score",
                "all_evaluation_metrics": None,
                "default_evaluation_metric": None
            },
            "F1 Score",
            None
        ),

        (
            "custom_metrics_valid_regression",
            {
                "ml_task_type": "Regression",
                "eval_metric": "MAE",
                "all_evaluation_metrics": None,
                "default_evaluation_metric": None
            },
            "MAE",
            None
        ),

    ])
    def test_eval_metric_checker(self, test_name, params, expected_result, expected_error):
        """Test eval_metric_checker validation"""
        if expected_error:
            with self.assertRaises(expected_error):
                eval_metric_checker(**params)
        else:
            result = eval_metric_checker(**params)
            self.assertEqual(result, expected_result)

    @parameterized.expand([
        # Valid cases
        (
            "none_value",
            None,
            None,
            None
        ),
        (
            "zero_value",
            0,
            0,
            None
        ),
        (
            "positive_integer",
            42,
            42,
            None
        ),

        # Invalid cases
        (
            "negative_integer",
            -1,
            None,
            ValueError
        ),
        (
            "float_value",
            42.0,
            None,
            ValueError
        )
    ])
    def test_random_state_checker(self, test_name, input_value, expected_result, expected_error):
        """Test random_state_checker validation"""
        if expected_error:
            with self.assertRaises(expected_error):
                random_state_checker(input_value)
        else:
            result = random_state_checker(input_value)
            self.assertEqual(result, expected_result)