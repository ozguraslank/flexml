from seaborn import load_dataset
import unittest
from flexml import Classification, Regression
import numpy as np

import warnings
warnings.filterwarnings("ignore")


class PerformanceTest(unittest.TestCase):
    """
    Test cases for the performance of the Classification class
    """

    df_class = load_dataset('diamonds')
    # Set seed for reproducibility
    np.random.seed(42)
    # Randomly select 20% of the data (excluding 'price') and set to NaN
    mask = np.random.rand(*df_class.shape) < 0.2
    mask[:, df_class.columns.get_loc('cut')] = False
    df_class.where(~mask, np.nan, inplace=True)
    fml_class = Classification(df_class, target_col='cut')
    
    df_regression = load_dataset('diamonds')
    # Randomly select 20% of the data (excluding 'price') and set to NaN
    mask = np.random.rand(*df_regression.shape) < 0.2
    mask[:, df_regression.columns.get_loc('price')] = False
    df_regression.where(~mask, np.nan, inplace=True)
    fml_regression = Regression(df_regression, target_col='price')

    def test_performance_classification(self):
        """
        Performance test for the Classification class
        """
        self.fml_class.start_experiment(experiment_size="wide",cv_method="holdout")

        # Calculate the average R2 score
        avg_accuracy = self.fml_class._model_stats_df["Accuracy"].mean()
        self.assertGreater(
            avg_accuracy, 
            0.55, 
            f"Average Accuracy score {avg_accuracy:.4f} is not greater than 0.55"
        )

    def test_performance_regression(self):
        """
        Performance test for the Classification class
        """
        self.fml_regression.start_experiment(experiment_size="wide",cv_method="holdout")

        # Calculate the average R2 score
        avg_r2 = self.fml_regression._model_stats_df["R2"].mean()
        self.assertGreater(
            avg_r2, 
            0.75, 
            f"Average R² score {avg_r2:.4f} is not greater than 0.75"
        )