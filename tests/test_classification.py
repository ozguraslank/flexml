from flexml.classification import Classification
from parameterized import parameterized
import unittest
import pandas as pd

class TestClassification(unittest.TestCase):
    df = pd.read_csv("tests/test_data/diabetes_classification.csv")
    
    @parameterized.expand(["quick", "wide"])
    def test_classification(self, exp_size: str, df: pd.DataFrame = df):
        try:
            classification_exp = Classification(
                data = df,
                target_col = "Outcome",
                experiment_size = exp_size,
                test_size = 0.25,
                random_state = 42
            )
        except Exception as e:
            raise Exception(f"An error occured while setting up {exp_size} classification, Error: {e}")
        
        try:
            classification_exp.start_experiment(top_n_models = 1)
        except Exception as e:
            raise Exception(f"An error occured while running {exp_size} classification experiment, Error: {e}")
        
        top_x_models = classification_exp.get_best_models(top_n_models = 3)
        if len(top_x_models) != 3:
            raise Exception(f"An error occured while retriving the best models in {exp_size} classification, expected 3, got {len(top_x_models)}")
        
        try:
            classification_exp.show_model_stats()
        except Exception as e:
            raise Exception(f"An error occured while showing models stats in {exp_size} classification, Error: {e}")
        
        try:
            classification_exp.tune_model()

            if classification_exp.tuned_model is None:
                raise Exception(f"An error occured while tuning the model in {exp_size} classification, tuned model is None")
            
            if classification_exp.tuned_model_score is None:
                raise Exception(f"An error occured while calculating the tuned model's score in {exp_size} classification, tuned model score is None")
        
        except Exception as e:
            raise Exception(f"An error occured while tuning the model in {exp_size} classification, Error: {e}")