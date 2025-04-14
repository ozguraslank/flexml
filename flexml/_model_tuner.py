import numpy as np
import pandas as pd
import optuna
import joblib
from joblib.parallel import BatchCompletionCallBack
from contextlib import contextmanager
from typing import Optional, Union
from time import time
from sklearn.model_selection import ParameterGrid, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from flexml.config import TUNING_METRIC_TRANSFORMATIONS
from flexml.logger import get_logger
from flexml.helpers import evaluate_model_perf
from copy import deepcopy
from tqdm import tqdm


class TqdmBatchCompletionCallback(BatchCompletionCallBack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        self.tqdm_object.update(n=self.batch_size)
        return super().__call__(*args, **kwargs)

@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


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

    logging_to_file: bool, (default=False)
        If True, the logs will be saved to a file in the current path, located in /logs/flexml_logs.log, Otherwise, it will not be saved
    """
    def __init__(
        self, 
        ml_problem_type: str,
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.DataFrame, np.ndarray],
        logging_to_file: bool = False
    ):
        """
        Parameters
        ----------
        ml_problem_type : str
            Type of the ML problem ('Classification' or 'Regression')

        X : pd.DataFrame
            The feature values of the dataset

        y : pd.DataFrame
            The target values of the dataset

        logging_to_file : bool, optional (default=False)
            Whether to log to a file
        """
        self.ml_problem_type = ml_problem_type.lower().capitalize()  # Normalize case
        self.X = X
        self.y = y

        self.logger = get_logger(__name__, "PROD", logging_to_file)

        self.eval_metrics_in_tuning_format = TUNING_METRIC_TRANSFORMATIONS.get(self.ml_problem_type)
        self.reverse_signed_eval_metrics = TUNING_METRIC_TRANSFORMATIONS.get("reverse_signed_eval_metrics")

        # Revise classification metrics for multi-class classification
        if self.ml_problem_type == "Classification" and self.y.nunique() > 2:
            self.eval_metrics_in_tuning_format['ROC-AUC'] = 'roc_auc_ovr'
            self.eval_metrics_in_tuning_format['Precision'] = 'precision_macro'
            self.eval_metrics_in_tuning_format['Recall'] = 'recall_macro'
            self.eval_metrics_in_tuning_format['F1 Score'] = 'f1_macro'

    def _param_grid_validator(
        self,
        model_available_params: dict,
        param_grid: dict,
        prefix_param_grid_flag: bool = True
    ) -> dict:
        """
        This method is used to validate the param_grid dictionary for the model

        Parameters
        ----------
        model_available_params : dict
            All params that model has

        param_grid : dict
            The dictionary that contains the hyperparameters and their possible values

        prefix_param_grid_flag : bool
            Indicates If param_grid keys will be modified to be suitable for Pipeline object, adds model__ prefix to the begining of them
        """
        param_amount = len(param_grid)
        if param_amount == 0:
            error_msg = "Error while validating the param_grid for the model. The param_grid should not be empty"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if prefix_param_grid_flag:
            param_grid = {f"model__{key}": value for key, value in param_grid.items()}
         
        # Check if all params that param_grid has are available in the model's params
        for param_name in param_grid.keys():
            if param_name not in model_available_params:
                error_msg = f"Error while validating the param_grid for the model. The '{param_name}' parameter is not available in the model's available params.\n \
                    Available params: {list(model_available_params)}"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
        return param_grid
    
    def _setup_tuning(
        self,
        tuning_method: str,
        model: Union[object, Pipeline],
        param_grid: dict,
        n_iter: Optional[int] = None,
        n_jobs: int = -1,
        prefix_param_grid_flag = True
    ):
        """
        Sets up the tuning process by creating the model_stats dictionary

        Parameters
        ----------
        tuning_method : str
            The tuning method that will be used for the optimization. It can be one of the following:
            
            * 'grid_search' for GridSearchCV (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
            
            * 'randomized_search' for RandomizedSearchCV (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
            
            * 'optuna' for Optuna (https://optuna.readthedocs.io/en/stable/)

        model : object or Pipeline
            The model or Pipeline object that will be used for tuning

        n_iter : int, optional (default=10)
            The number of iterations. The default is 10.
        
        n_jobs : int (default=-1)
            The number of parallel jobs to run. The default is -1.

        prefix_param_grid_flag : bool
            Indicates If param_grid keys will be modified to be suitable for Pipeline object, adds model__ prefix to the begining of them

        Returns
        -------
        model_stats: dict
            Dictionary including tuning information and model:

            * 'tuning_method': The tuning method that is used for the optimization
            
            * 'tuning_param_grid': The hyperparameter grid that is used for the optimization
            
            * 'n_iter': The number of iterations
            
            * 'n_jobs': The number of parallel jobs to run
            
            * 'tuned_model': The tuned model object
            
            * 'tuned_model_score': The evaluation metric score of the tuned model
            
            * 'tuned_model_evaluation_metric': The evaluation metric that is used to evaluate the tuned model
        """
        model_params = None
        
        if isinstance(model, Pipeline):
            model = model.named_steps['model']

        if "CatBoost" in model.__class__.__name__:
            model_params = model.get_all_params()
        else:
            model_params = model.get_params()
        
        if prefix_param_grid_flag:
            model_params = {f"model__{key}": value for key, value in model_params.items()}

        param_grid = self._param_grid_validator(
            model_available_params=model_params,
            param_grid=param_grid,
            prefix_param_grid_flag=prefix_param_grid_flag
        )

        model_stats = {
            "tuning_method": tuning_method,
            "tuning_param_grid": param_grid,
            "n_iter": n_iter,
            "n_jobs": n_jobs,
            "tuned_model": None,
            "tuned_model_score": None,
            "tuned_model_evaluation_metric": None
        }

        return model_stats
            
    def grid_search(
        self,
        pipeline: Pipeline,
        param_grid: dict,
        eval_metric: str,
        cv: list,
        n_jobs: int = -1,
        verbose: int = 0
    ) -> Optional[dict]:
        """
        Implements grid search hyperparameter optimization on the giveen machine learning model

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline object includes feature engineering and model object that will be tuned

        param_grid : dict
            The dictionary that contains the hyperparameters and their possible values

        eval_metric : str
            The evaluation metric that will be used to evaluate the model. It can be one of the following:
            
            * 'R2' for R^2 score
            
            * 'MAE' for Mean Absolute Error
            
            * 'MSE' for Mean Squared Error

            * 'RMSE' for Root Mean Squared Error

            * 'MAPE' for Mean Absolute Percentage Error
            
            * 'Accuracy' for Accuracy
            
            * 'Precision' for Precision
            
            * 'Recall' for Recall
            
            * 'F1 Score' for F1 score

        cv : list of tuples
            A list of (train_idx, test_idx) tuples where each tuple contains numpy arrays of indices
            for the training and test sets for that fold. For example:
            [(array([1,2,4,...]), array([0,3,6,...])), ...]

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
        model_stats = self._setup_tuning("GridSearchCV", pipeline, param_grid, n_iter=None, n_jobs=n_jobs)
        param_grid = model_stats['tuning_param_grid']
        
        try:
            t_start = time()
            
            # Calculate total fits
            total_params = len(ParameterGrid(param_grid))
            n_splits = len(cv)
            total_fits = total_params * n_splits

            # Create GridSearchCV object
            search = GridSearchCV(
                pipeline,
                param_grid,
                scoring=self.eval_metrics_in_tuning_format,
                refit=eval_metric,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose
            )
            
            # Fit with progress bar
            with tqdm_joblib(tqdm(
                total=total_fits,
                desc="INFO | Grid Search Progress",
                bar_format="{desc}: |{bar}| {percentage:.0f}%"
            )):
                search_result = search.fit(self.X, self.y)

            t_end = time()
            time_taken = round(t_end - t_start, 2)

            scores = {
                metric: (
                    -search_result.cv_results_[f'mean_test_{metric}'][search_result.best_index_]
                    if metric in self.reverse_signed_eval_metrics else
                    search_result.cv_results_[f'mean_test_{metric}'][search_result.best_index_]
                )
                for metric in list(self.eval_metrics_in_tuning_format.keys())
            }

            model_stats['tuned_model'] = search_result.best_estimator_.named_steps['model'] 
            mean_score = search_result.cv_results_[f'mean_test_{eval_metric}'][search_result.best_index_]
            model_stats['tuned_model_score'] = round(mean_score, 6)
            model_stats['model_perf'] = scores
            model_stats['time_taken_sec'] = time_taken
            model_stats['tuned_model_evaluation_metric'] = eval_metric
            return model_stats
        except Exception as e:
            self.logger.error(f"Error while tuning the model with GridSearchCV, Error: {e}")
            return None
    
    def randomized_search(
        self,
        pipeline: Pipeline,
        param_grid: dict,
        eval_metric: str,
        cv: list,
        n_iter: int = 10,
        n_jobs: int = -1,
        verbose: int = 0
    ) -> Optional[dict]:
        """
        Implements randomized search hyperparameter optimization on the giveen machine learning model

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline object includes feature engineering and model object that will be tuned

        param_grid : dict
            The dictionary that contains the hyperparameters and their possible values

        eval_metric : str
            The evaluation metric that will be used to evaluate the model. It can be one of the following:
            
            * 'R2' for R^2 score
            
            * 'MAE' for Mean Absolute Error
            
            * 'MSE' for Mean Squared Error

            * 'RMSE' for Root Mean Squared Error

            * 'MAPE' for Mean Absolute Percentage Error
            
            * 'Accuracy' for Accuracy
            
            * 'Precision' for Precision
            
            * 'Recall' for Recall
            
            * 'F1 Score' for F1 score

        cv : list of tuples
            A list of (train_idx, test_idx) tuples where each tuple contains numpy arrays of indices
            for the training and test sets for that fold. For example:
            [(array([1,2,4,...]), array([0,3,6,...])), ...]

        n_iter : int, optional (default=10)
            The number of trials. The default is 10

        n_jobs : int (default=-1)
            The number of parallel jobs to run. The default is -1
        
        Returns
        -------
        model_stats: dict
            Dictionary including tuning information and model:

            * 'tuning_method': The tuning method that is used for the optimization
            
            * 'tuning_param_grid': The hyperparameter grid that is used for the optimization
            
            * 'n_jobs': The number of parallel jobs to run
            
            * 'tuned_model': The tuned model object
            
            * 'tuned_model_score': The evaluation metric score of the tuned model
            
            * 'tuned_model_evaluation_metric': The evaluation metric that is used to evaluate the tuned model
        """
        model_stats = self._setup_tuning("randomized_search", pipeline, param_grid, n_iter=n_iter, n_jobs=n_jobs)
        param_grid = model_stats['tuning_param_grid']

        t_start = time()
        
        # Calculate total fits
        n_splits = len(cv)
        total_fits = n_iter * n_splits

        # Create RandomizedSearchCV object
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid, 
            n_iter=n_iter,
            scoring=self.eval_metrics_in_tuning_format, 
            refit=eval_metric,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        # Fit with progress bar
        with tqdm_joblib(tqdm(
            total=total_fits,
            desc="INFO | Randomized Search Progress",
            bar_format="{desc}: |{bar}| {percentage:.0f}%"
        )):
            search_result = search.fit(self.X, self.y)

        t_end = time()
        time_taken = round(t_end - t_start, 2)

        scores = {
            metric: (
                -search_result.cv_results_[f'mean_test_{metric}'][search_result.best_index_]
                if metric in self.reverse_signed_eval_metrics else
                search_result.cv_results_[f'mean_test_{metric}'][search_result.best_index_]
            )
            for metric in list(self.eval_metrics_in_tuning_format.keys())
        }

        model_stats['tuned_model'] = search_result.best_estimator_.named_steps['model']
        mean_score = search_result.cv_results_[f'mean_test_{eval_metric}'][search_result.best_index_]
        model_stats['tuned_model_score'] = round(mean_score, 6)
        model_stats['model_perf'] = scores
        model_stats['time_taken_sec'] = time_taken
        model_stats['tuned_model_evaluation_metric'] = eval_metric
        return model_stats

        
    def optuna_search(
        self,
        pipeline: Pipeline,
        param_grid: dict,
        eval_metric: str,
        cv: list,
        n_iter: int = 10,
        timeout: Optional[int] = None,
        n_jobs: int = -1,
        verbose: int = 0
    ) -> Optional[dict]:
        """
        Implements Optuna hyperparameter optimization on the given machine learning model

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline object includes feature engineering and model object that will be tuned

        param_grid : dict
            The dictionary that contains the hyperparameters and their possible values

        eval_metric : str
            The evaluation metric that will be used to evaluate the model. It can be one of the following:
            
            * 'R2' for R^2 score
            
            * 'MAE' for Mean Absolute Error
            
            * 'MSE' for Mean Squared Error

            * 'RMSE' for Root Mean Squared Error

            * 'MAPE' for Mean Absolute Percentage Error
            
            * 'Accuracy' for Accuracy
            
            * 'Precision' for Precision
            
            * 'Recall' for Recall
            
            * 'F1 Score' for F1 score

        cv : list of tuples
            A list of (train_idx, test_idx) tuples where each tuple contains numpy arrays of indices
            for the training and test sets for that fold. For example:
            [(array([1,2,4,...]), array([0,3,6,...])), ...]

        n_iter : int, optional (default=10)
            The number of trials. The default is 10

        timeout : int, optional (default=None)
            The timeout in seconds. The default is None

        n_jobs : int, optional (default=-1)
            The number of parallel jobs to run. The default is -1

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
        model_stats = self._setup_tuning("optuna", pipeline, param_grid, n_iter=n_iter, n_jobs=n_jobs, prefix_param_grid_flag=False)
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

        study_direction = "maximize" if eval_metric in ['R2', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'] else "minimize"

        def objective(trial):
            # Generate parameters for the trial
            params = pipeline.named_steps['model'].get_params()
            for param_name, param_values in param_grid.items():
                first_element = param_values[0]
                
                if isinstance(first_element, (str, bool)):
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
                elif isinstance(first_element, int):
                    params[param_name] = trial.suggest_int(param_name, param_values[0], param_values[-1])
                elif isinstance(first_element, float):
                    params[param_name] = trial.suggest_float(param_name, param_values[0], param_values[-1])
                else:
                    info_msg = f"{param_name} parameter is not added to tuning since its type is not supported by Optuna."
                    self.logger.info(info_msg)

            new_pipeline = Pipeline(steps=pipeline.steps[:-1] + [('model', clone(pipeline.named_steps['model']).set_params(**params))]) # Remove the old model from the pipeline and add the new one

            # Perform cross-validation and calculate the score
            scores = []
            for train_idx, test_idx in cv:
                X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

                new_pipeline.fit(X_train, y_train)

                if self.ml_problem_type == "Classification" and hasattr(new_pipeline, 'predict_proba'):
                    y_pred = new_pipeline.predict_proba(X_test)
                else:   
                    y_pred = new_pipeline.predict(X_test)

                # Evaluate performance
                scores.append(evaluate_model_perf(self.ml_problem_type, y_test, y_pred))

            # Calculate the mean score across all folds
            avg_metrics = {k: np.mean([m[k] if m[k] is not None else -1 for m in scores]) for k in scores[0]}
            mean_score = avg_metrics.get(eval_metric, float('inf'))

            # Update the best score and model
            if model_stats['tuned_model_score'] is None or (study_direction == "maximize" and mean_score > model_stats['tuned_model_score']) or (study_direction == "minimize" and mean_score < model_stats['tuned_model_score']):
                model_stats['tuned_model_score'] = round(mean_score, 6)
                model_stats['tuned_model'] = new_pipeline.named_steps['model']
                model_stats['model_perf'] = avg_metrics

            return mean_score
        
        try:
            # Perform Optuna optimization
            t_start = time()
            study = optuna.create_study(direction=study_direction)
            study.optimize(objective, n_trials=n_iter, timeout=timeout, n_jobs=n_jobs, show_progress_bar=True)
            t_end = time()

            # Update model stats
            model_stats['time_taken_sec'] = round(t_end - t_start, 2)
            return model_stats
        
        except Exception as e:
            self.logger.error(f"Error while tuning the model with Optuna, Error: {e}")
            return None