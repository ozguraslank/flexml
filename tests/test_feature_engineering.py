import unittest
import pandas as pd
import numpy as np
from flexml.classification import Classification

class TestClassification(unittest.TestCase):
    # categorical_imputation_methods = ["mode","constant"]
    # numerical_imputation_methods = ["mean","constant","median","mode"]
    # encoding_methods = ["label_encoder", "onehot_encoder"]

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


    def test_feature_engineering_with_inputs(self):
        """
        End-to-end test for feature engineering pipeline through Classification class
        """

        classification_exp = Classification(self.df, 
                            target_col='target',
                            drop_columns=['id'],
                            column_imputation_map={'status': 'constant','amount': 'constant'},
                            categorical_imputation_constant='test_constant',
                            numerical_imputation_constant=-1,
                            encoding_method_map={'category_default': 'ordinal_encoder', 'priority': 'onehot_encoder'},
                            ordinal_encode_map={'category_default': ['A', 'C', 'B']},
                            onehot_limit=3,
                            normalize_numerical='normalize_scaler'
                            )
        
        
        classification_exp.start_experiment()
        processed_data = classification_exp.data

        # Check if all columns are numerical, including target
        self.assertFalse(
        processed_data.select_dtypes(exclude=[np.number]).columns.tolist(),
        "Not all columns are numerical."
        )

        # Check for total number of columns
        self.assertEqual(
            processed_data.shape[1],
            9,
            "The processed data with inputs does not have 7 columns including the target."
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
        "Not all columns are numerical."
        )

        # Check for total number of columns
        self.assertEqual(
            processed_data.shape[1],
            8,
            "The default processed data does not have 8 columns including the target."
        )