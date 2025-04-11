import unittest
import numpy as np
from parameterized import parameterized
from sklearn.datasets import load_diabetes, load_breast_cancer
from flexml.regression import Regression
from flexml.classification import Classification
from flexml.helpers import get_cv_splits
from flexml.logger import get_logger
from flexml.config import get_ml_models

import warnings
warnings.filterwarnings("ignore")

class TestMLModels(unittest.TestCase):
    logger = get_logger(__name__, "TEST", logging_to_file=False)
    logger.setLevel("DEBUG")

    test_config = {
        'Regression': {
            'data': load_diabetes(as_frame=True)['frame'],
            'target_col': 'target',
            'exp_class': Regression,
            'models': get_ml_models(ml_task_type="Regression")['WIDE']
        },
        'Classification': {
            'data': load_breast_cancer(as_frame=True)['frame'],
            'target_col': 'target',
            'exp_class': Classification,
            'models': get_ml_models(ml_task_type="Classification")['WIDE']
        }
    }

    experiments = {}
    cv_splitters = {}
    
    for objective, config in test_config.items():
        exp = config['exp_class'](
            data=config['data'],
            target_col=config['target_col'],
            logging_to_file=False
        )
        experiments[objective] = exp
        
        cv_splitters[objective] = get_cv_splits(
            df=config['data'],
            cv_method="holdout",
            test_size=0.5 # Keeping test_size high to make the training faster
        )

    @parameterized.expand([
        (objective, model_pack['name'], model_pack['model'], model_pack['tuning_param_grid'])
        for objective, config in test_config.items()
        for model_pack in config['models']
    ])
    def test_ml_models(self, objective, model_name, model, model_tuning_params):
        exp = self.experiments[objective]
        cv_splitter = self.cv_splitters[objective]

        X, y = exp.X, exp.y
        train_idx = cv_splitter[0][0] # holdout validation returns in [(train_index, test_index)] format

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        
        model.fit(X_train, y_train)

        # If its classification problem
        if objective == 'Classification':
            predictions = model.predict_proba(X_train)
        else:
            predictions = model.predict(X_train)
        
        self.assertIsInstance(predictions, np.ndarray)

        try:
            exp.tune_model(
                model=model,
                tuning_method="randomized_search",
                param_grid=model_tuning_params,
                n_iter=3,
                n_folds=3,
                n_jobs=-1
            )

        except Exception as e:
            if 'Model leaderboard is empty!' in str(e):
                # Since we don't use the start_experiment() function, there will be no saved models and this error will be raised --
                # Because, we call _show_tuning_report when tune_model operation is done and that function calls get_best_models() function that calls __top_n_models_checker() where the error will be raised :)
                pass
            else:
                # Handle other exceptions
                error_msg = f"An error occurred while tuning {model_name} model with the following param_grid {model_tuning_params}. Error: {e}"
                self.logger.error(error_msg)                
                raise Exception(error_msg)