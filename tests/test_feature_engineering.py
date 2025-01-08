import unittest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from flexml.classification import Classification

import warnings
warnings.filterwarnings("ignore")


class TestFeatureEngineering(unittest.TestCase):
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

    for column in df.columns:
        if column not in ['id', 'target']:
            mask = np.random.random(n_rows) < 0.2
            df.loc[mask, column] = np.nan

    encoding_methods = ['label_encoder', 'onehot_encoder', 'ordinal_encoder']
    imputation_methods = ['mean', 'median', 'mode', 'constant', 'drop']
    normalization_methods = ['standard_scaler', 'minmax_scaler', 'robust_scaler', 'quantile_transformer', 'maxabs_scaler', 'normalize_scaler']

    def test_feature_engineering_with_inputs(self):
        """
        End-to-end test for feature engineering pipeline through Classification class
        """
        classification_exp = Classification(
                                self.df, 
                                target_col='target',
                                drop_columns=['id'],
                                column_imputation_map={'status': 'constant','amount': 'constant'},
                                categorical_imputation_constant='test_constant',
                                numerical_imputation_constant=-1,
                                encoding_method_map={'category_default': 'ordinal_encoder', 'priority': 'onehot_encoder'},
                                ordinal_encode_map={'category_default': ['A', 'C', 'B']},
                                onehot_limit=3,
                                normalize='normalize_scaler'
                            )
        
        
        classification_exp.start_experiment()
        processed_data = classification_exp.data

        # Check if all columns are numerical, including target
        self.assertFalse(
            processed_data.select_dtypes(exclude=[np.number]).columns.tolist(),
            "Not all columns are numerical"
        )

        # Check for total number of columns
        self.assertEqual(
            processed_data.shape[1],
            9,
            "The processed data with inputs does not have 7 columns including the target"
        )

    def test_feature_engineering_without_inputs(self):
        """
        End-to-end test for feature engineering pipeline through Classification class
        """
        classification_exp = Classification(self.df, target_col='target')
        
        classification_exp.start_experiment()
        processed_data = classification_exp.data
        print(processed_data)

        # Check if all columns are numerical, including target
        self.assertFalse(
            processed_data.select_dtypes(exclude=[np.number]).columns.tolist(),
            "Not all columns are numerical"
        )

        # Check for total number of columns
        self.assertEqual(
            processed_data.shape[1],
            8,
            "The default processed data does not have 8 columns including the target"
        )

    def test_feature_engineering_with_dynamic_inputs(self):
        """
        Dynamic end-to-end test for feature engineering pipeline through Classification class
        """
        # Nested loops for encoding, imputation, and normalization methods
        for encoding_method in self.encoding_methods:
            for imputation_method in self.imputation_methods:
                for normalization_method in self.normalization_methods:
                    encoding_method_map = {'category_default': encoding_method, 'priority': encoding_method}
                    ordinal_encode_map = None

                    # Handle specific cases for encoding methods
                    if encoding_method == 'ordinal_encoder':
                        ordinal_encode_map = {'priority': ['Low', 'Medium', 'High'], 'category_default':['A','C','B']}

                    # Distinguish between categorical and numerical imputation methods
                    if imputation_method in ['mode', 'constant', 'drop']:
                        column_imputation_map = {'status': imputation_method, 'amount': 'mean'}
                    elif imputation_method in ['mean', 'median']:
                        column_imputation_map = {'status': 'mode', 'amount': imputation_method}

                    with self.subTest(encoding_method=encoding_method, imputation_method=imputation_method, normalization_method=normalization_method):
                        classification_exp = Classification(
                            data=self.df, 
                            target_col='target',
                            drop_columns=['id'],
                            column_imputation_map=column_imputation_map,
                            categorical_imputation_constant='test_constant',
                            numerical_imputation_constant=-1,
                            encoding_method_map=encoding_method_map,
                            ordinal_encode_map=ordinal_encode_map,
                            onehot_limit=3,
                            normalize=normalization_method
                        )

                        # Prepare cross-validation splits
                        cv_splits = list(classification_exp._prepare_data(cv_method="kfold", n_folds=3, apply_feature_engineering=True)).copy()
                        preds = None

                        # Train-test split and logistic regression model
                        for train_idx, test_idx in cv_splits:
                            X_train, X_test = classification_exp.X.iloc[train_idx], classification_exp.X.iloc[test_idx]
                            y_train, y_test = classification_exp.y.iloc[train_idx], classification_exp.y.iloc[test_idx]

                            logistic = LogisticRegression(max_iter=500).fit(X_train, y_train)
                            preds = logistic.predict(X_test)

                        if preds is None:
                            raise Exception("Logistic regression failed to produce predictions")