import unittest
import pandas as pd
import numpy as np
from flexml._feature_engineer import FeatureEngineering
from sklearn.linear_model import LogisticRegression

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

    # This will create artificial null values within dataframe
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
        feature_exp = FeatureEngineering(
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
        
        feature_exp.setup()
        
        X_train, y_train = feature_exp.fit_transform()
        lr = LogisticRegression(max_iter=500).fit(X_train, y_train)

        # Check if all columns are numerical, including target
        self.assertFalse(
            X_train.select_dtypes(exclude=[np.number]).columns.tolist(),
            "Not all columns are numerical"
        )

        # Check if there are any null values
        self.assertFalse(
            X_train.isnull().any().any(),
            "There are null values in the processed data"
        )

    def test_feature_engineering_without_inputs(self):
        """
        End-to-end test for feature engineering pipeline through Classification class
        """
        feature_exp = FeatureEngineering(self.df, target_col='target')
        feature_exp.setup()
        
        X_train, y_train = feature_exp.fit_transform()
        lr = LogisticRegression(max_iter=500).fit(X_train, y_train)

        # Check if all columns are numerical, including target
        self.assertFalse(
            X_train.select_dtypes(exclude=[np.number]).columns.tolist(),
            "Not all columns are numerical"
        )

        # Check if there are any null values
        self.assertFalse(
            X_train.isnull().any().any(),
            "There are null values in the processed data"
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
                        feature_test = FeatureEngineering(
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
                        feature_test.setup()
                        
                        X_train, y_train = feature_test.fit_transform()
                        lr = LogisticRegression(max_iter=500).fit(X_train, y_train)

                        # Check if all columns are numerical, including target
                        self.assertFalse(
                            X_train.select_dtypes(exclude=[np.number]).columns.tolist(),
                            f"Not all columns are numerical. Failed parameters are: "
                            f"Encoding method: {encoding_method}, "
                            f"Imputation method: {imputation_method}, "
                            f"Normalization method: {normalization_method}"
                        )

                        # Check if there are any null values
                        self.assertFalse(
                            X_train.isnull().any().any(),
                            f"There are null values in the processed data. Failed parameters are: "
                            f"Encoding method: {encoding_method}, "
                            f"Imputation method: {imputation_method}, "
                            f"Normalization method: {normalization_method}"
                        )