import numpy as np
import pandas as pd
import optuna
from typing import Optional
from time import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score)

from flexml.logger.logger import get_logger
from flexml.helpers import eval_metric_checker


class ModelTuner:
    """
    Implements hyperparameter tuning on the machine learning models with the desired tuning method from the following:

    * 'grid_search' for GridSearchCV (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
        Note that GridSearch optimization may take too long to finish since It tries all the possible combinations of the parameters

    * 'randomized_search' for RandomizedSearchCV (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
            
    * 'optuna' for Optuna (https://optuna.readthedocs.io/en/stable/)

    Parameters
    ----------
    ml_problem_type : str
        The type of the machine learning problem. It can be one of the following:
        
        * 'Classification' for classification problems
        
        * 'Regression' for regression problems

    X_train : pd.DataFrame
        The training set features
    
    X_test : pd.DataFrame
        The test set features

    y_train : pd.DataFrame
        The training set target values

    y_test : pd.DataFrame
        The test set target values

    logging_to_file: bool, (default=False)
        If True, the logs will be saved to a file in the current path, located in /logs/flexml_logs.log, Otherwise, it will not be saved
    """
    def __init__(self, 
                 ml_problem_type: str,
                 X_train: pd.DataFrame, 
                 X_test: pd.DataFrame, 
                 y_train: pd.DataFrame, 
                 y_test: pd.DataFrame,
                 logging_to_file: bool = False):
        self.ml_problem_type = ml_problem_type.lower().capitalize() # Fix ml_problem_type's format in just case, It should be in the following format: 'Example'
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.logger = get_logger(__name__, "PROD", logging_to_file)

    @staticmethod
    def __eval_metric_revieser(eval_metric: str) -> str:
        """
        Scikit-learn based hyperparameter optimization methods (GridSearch & Randomized Search) require spesific namings for evaluation metrics

        This method is used to revise the evaluation metric name for the optimization process

        Parameters
        ----------
        eval_metric : str
            The evaluation metric

        Returns
        -------
        str
            The revised evaluation metric name. e.g. 'R2' to 'r2, 'Accuracy' to 'accuracy', 'F1 Score' to 'f1_weighted' etc.
        """
        return eval_metric.lower() if eval_metric != 'F1 Score' else 'f1_weighted'
 
    def _param_grid_validator(self,
                              model_available_params: dict,
                              param_grid: dict) -> dict:
        """
        This method is used to validate the param_grid dictionary for the model

        Parameters
        ----------
        model_available_params : dict
            All params that model has

        param_grid : dict
            The dictionary that contains the hyperparameters and their possible values
        """
        param_amount = len(param_grid)
        if param_amount == 0:
            error_msg = "Error while validating the param_grid for the model. The param_grid should not be empty"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
         
        # Check if all params that param_grid has are available in the model's params
        for param_name in param_grid.keys():
            if param_name not in model_available_params:
                error_msg = f"Error while validating the param_grid for the model. The '{param_name}' parameter is not available in the model's available params.\n \
                    Available params: {list(model_available_params)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
        return param_grid
    
    def _setup_tuning(self,
                      tuning_method: str,
                      model: object,
                      param_grid: dict,
                      n_iter: Optional[int] = None,
                      cv: int = 3,
                      n_jobs: int = -1):
        """
        Sets up the tuning process by creating the model_stats dictionary

        Parameters
        ----------
        tuning_method : str
            The tuning method that will be used for the optimization. It can be one of the following:
            
            * 'grid_search' for GridSearchCV (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
            
            * 'randomized_search' for RandomizedSearchCV (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
            
            * 'optuna' for Optuna (https://optuna.readthedocs.io/en/stable/)

        model : object
            The model object that will be tuned.

        n_iter : int, optional (default=10)
            The number of iterations. The default is 10.

        cv : int (default=3)
            The number of cross-validation splits. The default is 3.
        
        n_jobs : int (default=-1)
            The number of parallel jobs to run. The default is -1.

        Returns
        -------
        model_stats: dict
            Dictionary including tuning information and model:

            * 'tuning_method': The tuning method that is used for the optimization
            
            * 'tuning_param_grid': The hyperparameter grid that is used for the optimization
            
            * 'n_iter': The number of iterations

            * 'cv': The number of cross-validation splits
            
            * 'n_jobs': The number of parallel jobs to run
            
            * 'tuned_model': The tuned model object
            
            * 'tuned_model_score': The evaluation metric score of the tuned model
            
            * 'tuned_model_evaluation_metric': The evaluation metric that is used to evaluate the tuned model
        """
        model_params = None

        if "CatBoost" in model.__class__.__name__:
            model_params = model.get_all_params()
        else:
            model_params = model.get_params()

        param_grid = self._param_grid_validator(model_params, param_grid)
        model_stats = {
            "tuning_method": tuning_method,
            "tuning_param_grid": param_grid,
            "n_iter": n_iter,
            "cv": cv,
            "n_jobs": n_jobs,
            "tuned_model": None,
            "tuned_model_score": None,
            "tuned_model_evaluation_metric": None
        }

        return model_stats
    
    def _model_evaluator(self,
                         model: object,
                         eval_metric: str):
        """
        Evaluates the model with the given evaluation metric by using the test set

        Parameters
        ----------
        model : object
            The model object that will be evaluated.

        eval_metric : str
            The evaluation metric that will be used to evaluate the model. It can be one of the following:
            
            * 'R2' for R^2 score
            
            * 'MAE' for Mean Absolute Error
            
            * 'MSE' for Mean Squared Error
            
            * 'Accuracy' for Accuracy
            
            * 'Precision' for Precision
            
            * 'Recall' for Recall
            
            * 'F1 Score' for F1 score
        """
        eval_metric = eval_metric_checker(self.ml_problem_type, eval_metric)
        
        if eval_metric == 'R2':
            return round(r2_score(self.y_test, model.predict(self.X_test)), 6)
        elif eval_metric == 'MAE':
            return round(mean_absolute_error(self.y_test, model.predict(self.X_test)), 6)
        elif eval_metric == 'MSE':
            return round(mean_squared_error(self.y_test, model.predict(self.X_test)), 6)
        elif eval_metric == 'Accuracy':
            return round(accuracy_score(self.y_test, model.predict(self.X_test)), 6)
        elif eval_metric == 'Precision':
            return round(precision_score(self.y_test, model.predict(self.X_test)), 6)
        elif eval_metric == 'Recall':
            return round(recall_score(self.y_test, model.predict(self.X_test)), 6)
        elif eval_metric == 'F1 Score':
            return round(f1_score(self.y_test, model.predict(self.X_test)), 6)
        else:
            error_msg = "Error while evaluating the current model during the model tuning process. The eval_metric should be one of the following: 'R2', 'MAE', 'MSE', 'Accuracy', 'Precision', 'Recall', 'F1 Score'"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
    def grid_search(self,
                    model: object,
                    param_grid: dict,
                    eval_metric: str,
                    cv: int = 3,
                    n_jobs: int = -1,
                    verbose: int = 0) -> Optional[dict]:
        """
        Implements grid search hyperparameter optimization on the giveen machine learning model

        Parameters
        ----------
        model : object
            The model object that will be tuned.

        param_grid : dict
            The dictionary that contains the hyperparameters and their possible values.

        eval_metric : str
            The evaluation metric that will be used to evaluate the model. It can be one of the following:
            
            * 'R2' for R^2 score
            
            * 'MAE' for Mean Absolute Error
            
            * 'MSE' for Mean Squared Error
            
            * 'Accuracy' for Accuracy
            
            * 'Precision' for Precision
            
            * 'Recall' for Recall
            
            * 'F1 Score' for F1 score

        cv : int (default=3)
            The number of cross-validation splits. The default is 3.

        n_jobs : int (default=-1)
            The number of parallel jobs to run. The default is -1.

        verbose: int (default = 0)
            The verbosity level of the tuning process. If It's set to 0, no logs will be shown during the tuning process. Otherwise, the logs will be shown based on the value of the verbose parameter:
            
            * 1 : the computation time for each fold and parameter candidate is displayed

            * 2 : the score is also displayed

            * 3 : the fold and candidate parameter indexes are also displayed together with the starting time of the computation

        Returns
        -------
        model_stats: dict
            Dictionary including tuning information and model:

            * 'tuning_method': The tuning method that is used for the optimization
            
            * 'tuning_param_grid': The hyperparameter grid that is used for the optimization
            
            * 'cv': The number of cross-validation splits
            
            * 'n_jobs': The number of parallel jobs to run
            
            * 'tuned_model': The tuned model object
            
            * 'tuned_model_score': The evaluation metric score of the tuned model
            
            * 'tuned_model_evaluation_metric': The evaluation metric that is used to evaluate the tuned model
        """
        model_stats = self._setup_tuning("GridSearchCV", model, param_grid, n_iter=None, cv=cv, n_jobs=n_jobs)
        param_grid = model_stats['tuning_param_grid']
        scoring_eval_metric = self.__eval_metric_revieser(eval_metric)
        
        try:
            t_start = time()
            search_result = GridSearchCV(model, param_grid, scoring=scoring_eval_metric, cv=cv, n_jobs=n_jobs, verbose=verbose).fit(self.X_train, self.y_train)
            t_end = time()
            time_taken = round(t_end - t_start, 2)

            model_stats['tuned_model'] = search_result.best_estimator_
            model_stats['tuned_model_score'] = round(self._model_evaluator(search_result.best_estimator_, eval_metric), 6)
            model_stats['time_taken_sec'] = time_taken
            model_stats['tuned_model_evaluation_metric'] = eval_metric
            return model_stats
        
        except Exception as e:
            self.logger.error(f"Error while tuning the model with GridSearchCV, Error: {e}")
            return None
    
    def random_search(self,
                      model: object,
                      param_grid: dict,
                      eval_metric: str,
                      n_iter: int = 10,
                      cv: int = 3,
                      n_jobs: int = -1,
                      verbose: int = 0) -> Optional[dict]:
        """
        Implements random search hyperparameter optimization on the giveen machine learning model

        Parameters
        ----------
        model : object
            The model object that will be tuned.

        param_grid : dict
            The dictionary that contains the hyperparameters and their possible values.

        eval_metric : str
            The evaluation metric that will be used to evaluate the model. It can be one of the following:
            
            * 'R2' for R^2 score
            
            * 'MAE' for Mean Absolute Error
            
            * 'MSE' for Mean Squared Error
            
            * 'Accuracy' for Accuracy
            
            * 'Precision' for Precision
            
            * 'Recall' for Recall
            
            * 'F1 Score' for F1 score

        n_iter : int, optional (default=10)
            The number of trials. The default is 10.

        cv : int (default=3)
            The number of cross-validation splits. The default is 3.

        n_jobs : int (default=-1)
            The number of parallel jobs to run. The default is -1.
        
        Returns
        -------
        model_stats: dict
            Dictionary including tuning information and model:

            * 'tuning_method': The tuning method that is used for the optimization
            
            * 'tuning_param_grid': The hyperparameter grid that is used for the optimization
            
            * 'cv': The number of cross-validation splits
            
            * 'n_jobs': The number of parallel jobs to run
            
            * 'tuned_model': The tuned model object
            
            * 'tuned_model_score': The evaluation metric score of the tuned model
            
            * 'tuned_model_evaluation_metric': The evaluation metric that is used to evaluate the tuned model
        """
        model_stats = self._setup_tuning("randomized_search", model, param_grid, n_iter=n_iter, cv=cv, n_jobs=n_jobs)
        param_grid = model_stats['tuning_param_grid']
        scoring_eval_metric = self.__eval_metric_revieser(eval_metric)

        try:
            t_start = time()
            search_result = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=n_iter, scoring=scoring_eval_metric, cv=cv, n_jobs=n_jobs, verbose=verbose).fit(self.X_train, self.y_train)
            t_end = time()
            time_taken = round(t_end - t_start, 2)

            model_stats['tuned_model'] = search_result.best_estimator_
            model_stats['tuned_model_score'] = round(self._model_evaluator(search_result.best_estimator_, eval_metric), 6)
            model_stats['time_taken_sec'] = time_taken
            model_stats['tuned_model_evaluation_metric'] = eval_metric
            return model_stats
        
        except Exception as e:
            self.logger.error(f"Error while tuning the model with RandomizedSearchCV, Error: {e}")
            return None
        
    def optuna_search(self,
               model: object,
               param_grid: dict,
               eval_metric: str,
               n_iter: int = 10,
               timeout: Optional[int] = None,
               n_jobs: int = -1,
               verbose: int = 0) -> Optional[dict]:
        """
        Implements Optuna hyperparameter optimization on the giveen machine learning model

        Parameters
        ----------
        model : object
            The model object that will be tuned.

        param_grid : dict
            The dictionary that contains the hyperparameters and their possible values.

        eval_metric : str
            The evaluation metric that will be used to evaluate the model. It can be one of the following:
            
            * 'R2' for R^2 score
            
            * 'MAE' for Mean Absolute Error
            
            * 'MSE' for Mean Squared Error
            
            * 'Accuracy' for Accuracy
            
            * 'Precision' for Precision
            
            * 'Recall' for Recall
            
            * 'F1 Score' for F1 score

        n_iter : int, optional (default=100)
            The number of trials. The default is 100.

        timeout : int, optional (default=None)
            The timeout in seconds. The default is None.

        n_jobs : int, optional (default=-1)
            The number of parallel jobs to run. The default is -1.

        verbose: int (default = 0)
            The verbosity level of the tuning process. If It's set to 0, no logs will be shown during the tuning process. Otherwise, the logs will be shown based on the value of the verbose parameter:

            * DEBUG (Equals to 4): Most detailed logging (prints almost everything)

            * INFO (Equals to 3): Standard informational output

            * WARNING (Equals to 2): Only warnings and errors

            * ERROR (Equals to 1): Only error messages

            * CRITICAL (Equals to 0): Only critical errors

        Returns
        -------
        model_stats: dict
            Dictionary including tuning information and model:

            * 'tuning_method': The tuning method that is used for the optimization
            
            * 'tuning_param_grid': The hyperparameter grid that is used for the optimization
            
            * 'cv': The number of cross-validation splits
            
            * 'n_jobs': The number of parallel jobs to run
            
            * 'tuned_model': The tuned model object
            
            * 'tuned_model_score': The evaluation metric score of the tuned model
            
            * 'tuned_model_evaluation_metric': The evaluation metric that is used to evaluate the tuned model
        """
        model_stats = self._setup_tuning("optuna", model, param_grid, n_iter=n_iter, n_jobs=n_jobs)
        param_grid = model_stats['tuning_param_grid']

        # Set verbosity levels
        if verbose == 0:
            optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        elif verbose == 1:
            optuna.logging.set_verbosity(optuna.logging.ERROR)
        elif verbose == 2:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        elif verbose == 3:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        elif verbose == 4:
            optuna.logging.set_verbosity(optuna.logging.DEBUG)

        study_direction = "maximize" if eval_metric in ['R2', 'Accuracy', 'Precision', 'Recall', 'F1 Score'] else "minimize"

        def objective(trial):
            """
            Brief explanation of the objective function usage here:

            * The objective function is used to optimize the hyperparameters of the model with Optuna
            * It's called in each trial and returns the evaluation metric score of the model with the current hyperparameters
            
            * In our scenario, we have to make the param grid dynamic for every model, so that:
                * We have to get the first element of the param_values to understand the data type of the hyperparameter
                * Then, we have to use the appropriate Optuna function to get the hyperparameter value for the current trial
            """
            params = {}
            for param_name, param_values in param_grid.items():
                first_element = param_values[0]

                if isinstance(first_element, str) or isinstance(first_element, bool):
                    param_value = trial.suggest_categorical(param_name, param_values)
                    params[param_name] = param_value

                elif isinstance(first_element, int):
                    param_value = trial.suggest_int(param_name, first_element, param_values[len(param_values) - 1])
                    params[param_name] = param_value

                elif isinstance(first_element, float):
                    param_value = trial.suggest_float(param_name, first_element, param_values[len(param_values) - 1])
                    params[param_name] = param_value

                # TODO: Other types can be added too, e.g. suggest_loguniform, suggest_uniform, suggest_discrete_uniform
                else:
                    info_msg = f"{param_name} parameter is not added to tuning process since It's data type is not supported for Optuna tuning\n \
                                Please use one of the following data types in your params: 'str', 'bool', 'int', 'float'. Instead of {type(first_element)}"
                    self.logger.info(info_msg)
            
            test_model = type(model)()
            test_model.set_params(**params)
            test_model.fit(self.X_train, self.y_train)
            
            score = self._model_evaluator(test_model, eval_metric)
            
            # Update the best score and best hyperparameters If the current score is better than the best one
            if model_stats['tuned_model_score'] is None or score > model_stats['tuned_model_score']:
                model_stats['tuned_model_score'] = round(score, 6)
                model_stats['tuned_model'] = test_model

            return score
        
        try:
            t_start = time()
            study = optuna.create_study(direction=study_direction)
            study.optimize(objective, n_trials=n_iter, timeout=timeout, n_jobs=n_jobs)
            t_end = time()
            time_taken = round(t_end - t_start, 2)
            model_stats['time_taken_sec'] = time_taken
            return model_stats
        
        except Exception as e:
            self.logger.error(f"Error while tuning the model with Optuna, Error: {e}")
            return None