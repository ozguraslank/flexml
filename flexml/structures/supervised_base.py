import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from typing import Union, Optional
from tqdm import tqdm
from IPython import get_ipython
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score)

from flexml.config.supervised_config import ML_MODELS, EVALUATION_METRICS
from flexml.logger.logger import get_logger
from flexml._model_tuner import ModelTuner


class SupervisedBase:
    """
    Base class for Supervised tasks (regression & classification)

    Parameters
    ----------
    data : pd.DataFrame
        The input data for the model training process.
    
    target_col : str
        The target column name in the data.

    logging_to_file: bool, (default=False)
        If True, the logs will be saved to a file in the current path, located in /logs/flexml_logs.log, Otherwise, it will not be saved.
    """
    def __init__(self,
                 data: pd.DataFrame,
                 target_col: str,
                 logging_to_file: str = False):
        self.data = data
        self.target_col = target_col
        self.logging_to_file = logging_to_file
        self.__ML_MODELS = []
        self.__ML_TASK_TYPE = "Regression" if "Regression" in self.__class__.__name__ else "Classification"
        self.__DEFAULT_EVALUATION_METRIC = EVALUATION_METRICS[self.__ML_TASK_TYPE]["DEFAULT"]
        self.__ALL_EVALUATION_METRICS = EVALUATION_METRICS[self.__ML_TASK_TYPE]["ALL"]

        # Logger to log app activities (Logs are stored in flexml/logs/log.log file)
        self.__logger = get_logger(__name__, "PROD", self.logging_to_file)

        # Data and ML model preparation stage
        self.__validate_data()
        self.feature_names = self.data.drop(columns=[self.target_col]).columns
        self.__model_training_info = []
        self.__model_stats_df = None
        self.__sorted_model_stats_df = None
        self.__data_is_prepared = False # Since _prepare_data() is going to be called in the start_experiment() method, data shouldn't be prepared again when start_experiment() is called again to test other scenarios
        self.__current_random_state = None # Keep random_state information to check if the data is prepared with the same random_state, If not, prepare the data again

    def __repr__(self):
        return f"SupervisedBase(\ndata={self.data.head()},\ntarget_col={self.target_col},\nlogging_to_file={self.logging_to_file})"
    
    def __validate_data(self):
        """
        Validates the input data given while initializing the Regression Class
        """
        # Data Overview validation
        if not isinstance(self.data, pd.DataFrame):
            error_msg = f"Dataframe should be a pandas DataFrame, but you've passed {type(self.data)}"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.data.select_dtypes(include=[np.number]).shape[1] != self.data.shape[1]:
            error_msg = "Dataframe should include only numeric values, did you forget to encode the categorical variables?"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.data.shape[0] == 0:
            error_msg = "Dataframe should include at least one row, is your dataframe empty?"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.data.shape[1] <= 1:
            error_msg = "Dataframe should include at least two columns, consider taking a look at your dataframe"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Target Column validation
        if self.target_col not in self.data.columns:
            error_msg = f"Target column '{self.target_col}' is not found in the data"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.data[self.target_col].isnull().sum() > 0:
            error_msg = "Target column should not include null values"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
    def _prepare_data(self, test_size: float = 0.25, random_state: int = 42):
        """
        Prepares the data for the model training process

        Parameters
        ----------
        test_size : float, (default=0.25)
            The size of the test data in the train-test split process

        random_state : int, (default=42)
            The random state value for the train-test split process
        """
        try:
            self.test_size = test_size
            self.__train_test_split(test_size, random_state)
            self.__data_is_prepared = True
            self.__logger.info("[PROCESS] Data is prepared")

        except Exception as e:
            error_msg = f"An error occurred while preparing the data: {str(e)}"
            self.__logger.error(error_msg)
            raise Exception(error_msg)
        
    def __prepare_models(self):
        """
        Prepares the models based on the selected experiment size ('quick' or 'wide')
        """
        if not isinstance(self.experiment_size, str):
            error_msg = f"experiment_size expected to be a string, got {type(self.experiment_size)}"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)

        if self.experiment_size not in ['quick', 'wide']:
            error_msg = f"experiment_size expected to be either 'quick' or 'wide', got {self.experiment_size}"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.__ML_MODELS = ML_MODELS.get(self.__ML_TASK_TYPE).get(self.experiment_size.upper())
        
    def __train_test_split(self, test_size: float, random_state: int) -> list[np.ndarray]:
        """
        Splits the data into train and test.
        Uses scikit-learn's train_test_split function, for more information, visit https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

        Parameters
        ----------
        test_size : float
            The size of the test data in the train-test split process

        random_state : int
            The random state value for shuffling the data before splitting

        Returns
        -------
        list[np.ndarray]
            A list of arrays containing the train and test data.
        """
        try:   
            X = self.data.drop(columns=[self.target_col])
            y = self.data[self.target_col]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        except Exception as e:
            error_msg = f"An error occurred while splitting the data into train and test: {str(e)}"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
    def __eval_metric_checker(self, eval_metric: Optional[str] = None) -> str:
        """
        Since eval_metric setting and validation is a common process for both Regression and Classification tasks...
        this method is used to set and validate the evaluation metric.

        Parameters
        ----------
        eval_metric : str
            The evaluation metric to use for model evaluation.
        
        Returns
        -------
        str
            The evaluation metric to use for model evaluation for the current task (Regression or Classification)
        """
        if eval_metric is None: # If the user passed nothing, the default evaluation metric will be used ('r2' for Regression, 'accuracy' for Classification)
            return self.__DEFAULT_EVALUATION_METRIC
        
        if not isinstance(eval_metric, str):
            error_msg = f"eval_metric expected to be a string, got {type(eval_metric)}"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        if eval_metric not in self.__ALL_EVALUATION_METRICS:
            error_msg = f"{eval_metric} is not a valid evaluation metric for {self.__ML_TASK_TYPE}, expected one of the following: {self.__ALL_EVALUATION_METRICS}"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        return eval_metric
    
    def __top_n_models_checker(self, top_n_models: Optional[int]) -> int:
        """
        Since top_n_models is a common process for both Regression
        """
        if top_n_models is None:
            return 1
        
        if not isinstance(top_n_models, int):
            error_msg = f"top_n_models expected to be an integer, got {type(top_n_models)}"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        if top_n_models < 1 or top_n_models > len(self.__ML_MODELS):
            error_msg = f"Invalid top_n_models value. Expected a value between 1 and {len(self.__ML_MODELS)}, got {top_n_models}"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        return top_n_models
    
    def __evaluate_model_perf(self, y_test, y_pred):
        """
        Evaluates how good are the predictions by comparing them with the actual values, returns regression evaluation scores

        Parameters
        ----------
        y_test : np.ndarray
            The actual values of the target column.
        
        y_pred : np.ndarray
            The predicted values of the target column.
        
        Returns
        -------
        dict
            A dictionary containing the evaluation metric of the current task
                
                * r2, mae, mse, rmse for Regression tasks

                * accuracy, precision, recall, f1_score for Classification tasks
        """

        if self.__ML_TASK_TYPE == "Regression":
            r2 = round(r2_score(y_test, y_pred), 4)
            mae = round(mean_absolute_error(y_test, y_pred), 4)
            mse = round(mean_squared_error(y_test, y_pred), 4)
            rmse = round(np.sqrt(mse), 4)
            return {
                "r2": r2,
                "mae": mae,
                "mse": mse,
                "rmse": rmse
            }
        
        elif self.__ML_TASK_TYPE == "Classification":
            accuracy = round(accuracy_score(y_test, y_pred), 4)
            precision = round(precision_score(y_test, y_pred, average='weighted'), 4)
            recall = round(recall_score(y_test, y_pred, average='weighted'), 4)
            f1 = round(f1_score(y_test, y_pred, average='weighted'), 4)
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
        
        else:
            error_msg = f"Unsupported task type, only 'Regression' and 'Classification' tasks are supported, got {self.__ML_TASK_TYPE}"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
    
    def start_experiment(self,
                        experiment_size: str = 'quick',
                        test_size: float = 0.25,
                        eval_metric: Optional[str] = None,
                        random_state: int = 42):
        """
        Trains machine learning algorithms and evaluates them based on the specified evaluation metric
        
        Parameters
        ----------
        experiment_size : str, (default='quick')
            The size of the experiment to run. It can be 'quick' or 'wide'
            
            * If It's selected 'quick', less amount of machine learning models will be used to get quick results
            
            * If It's selected 'wide', wide range of machine learning models will be used to get more comprehensive results

            You can take a look at the models in the library at config/ml_models.py

        test_size : float, (default=0.25)
            The size of the test data in the train-test split process.

        eval_metric : str (default='r2' for Regression, 'accuracy' for Classification)
            The evaluation metric to use for model evaluation.

        random_state : int, (default=42)
            The random state value for the train-test split process
            
            For more info, visit https://scikit-learn.org/stable/glossary.html#term-random_state
        """
        
        self.eval_metric = self.__eval_metric_checker(eval_metric)
        self.experiment_size = experiment_size
        self.test_size = test_size
        self.random_state = random_state
        self.__model_training_info = [] # Reset the model training info before starting the experiment
        self.__model_stats_df = None    # Reset the model stats DataFrame before starting the experiment

        if not self.__data_is_prepared or (self.__current_random_state and self.__current_random_state != random_state):
            self.__current_random_state = random_state
            self._prepare_data(test_size, random_state)

        self.__prepare_models()

        self.__logger.info("[PROCESS] Training the ML models")

        for model_idx in tqdm(range(len(self.__ML_MODELS))):
            model_info = self.__ML_MODELS[model_idx]
            model_name = model_info['name']
            model = model_info['model']
            try:
                t_start = time()
                model.fit(self.X_train, self.y_train)
                t_end = time()
                time_taken = round(t_end - t_start, 2)
                y_pred = model.predict(self.X_test)
                model_perf = self.__evaluate_model_perf(self.y_test, y_pred)

                self.__model_training_info.append({
                    model_name: {
                        "model": model,
                        "model_stats": {
                            "model_name": model_name,
                            **model_perf,
                            "Time Taken (sec)": time_taken}
                    }
                })

            except Exception as e:
                self.__logger.error(f"An error occured while training {model_name}: {str(e)}")

        self.__logger.info("[PROCESS] Model training is finished!")
        self.get_best_models(eval_metric)
        self.show_model_stats(eval_metric)

    def get_model_by_name(self, model_name: str) -> object:
        """
        Returns the model object by the given model name

        Parameters
        ----------
        model_name : str
            The name of the model to retrieve.

        Returns
        -------
        object
            The model object with the given model name
        """

        for model_info in self.__model_training_info:
            if model_name in model_info.keys():
                return model_info[model_name]["model"]
        
        error_msg = f"{model_name} is not found in the trained models, expected one of the following:\n{[list(model_info.keys())[0] for model_info in self.__model_training_info]}"
        self.__logger.error(error_msg)
        raise ValueError(error_msg)


    def get_best_models(self, eval_metric: Optional[str] = None, top_n_models: int = 1) -> Union[object, list[object]]:
        """
        Returns the top n models based on the evaluation metric.

        Parameters
        ----------
        top_n_models : int
            The number of top models to select based on the evaluation metric.
        eval_metric : str (default='r2 for Regression, 'accuracy' for Classification)
            The evaluation metric to use for model evaluation:
                
                * r2, mae, mse, rmse for Regression tasks

                * accuracy, precision, recall, f1_score for Classification tasks
        Returns
        -------
        object or list[object]
            Single or a list of top n models based on the evaluation metric.
        """
        if len(self.__model_training_info) == 0:
            error_msg = "There is no model performance data to sort!"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        top_n_models = self.__top_n_models_checker(top_n_models)

        if eval_metric is not None:
            eval_metric = self.__eval_metric_checker(eval_metric)
        else: # If the user doesn't pass a eval_metric, get the evaluation metric passed to the start_experiment function
            eval_metric = self.eval_metric
        
        model_stats = []
        best_models = []
        
        for model_pack in self.__model_training_info:
            for model_name, model_data in model_pack.items():
                model_stats.append(model_data["model_stats"])
    
        self.__model_stats_df = pd.DataFrame(model_stats)
        self.__sorted_model_stats_df = self.__sort_models(eval_metric)

        for i in range(top_n_models):
            searched_model_name = self.__sorted_model_stats_df.iloc[i]["model_name"]
            for model_info in self.__model_training_info:
                model_name = list(model_info.keys())[0]
                if model_name == searched_model_name:
                    best_models.append(model_info[model_name]["model"])
        
        if len(best_models) == 1:
            return best_models[0]
        
        return best_models

    def __sort_models(self, eval_metric: Optional[str] = None):
        """
        Sorts the models based on the evaluation metric.

        Parameters
        ----------
        eval_metric : str (default='r2')
            The evaluation metric to use for model evaluation (e.g. 'r2', 'mae', 'mse', 'rmse')

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the sorted model statistics according to the desired eval_metric.
        """
        if len(self.__model_stats_df) == 0:
            error_msg = "There is no model performance data to sort!"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        eval_metric = self.__eval_metric_checker(eval_metric)
        
        # Since lower is better for mae, mse and rmse in Regression tasks, they should be sorted in ascending order
        if self.__ML_TASK_TYPE == "Regression" and eval_metric in ['mae', 'mse', 'rmse']:
            return self.__model_stats_df.sort_values(by=eval_metric, ascending=True).reset_index(drop = True)
        else:
            return self.__model_stats_df.sort_values(by=eval_metric, ascending=False).reset_index(drop = True)

    def show_model_stats(self, eval_metric: Optional[str] = None):
        """
        Sorts and shows the model statistics table based on the evaluation metric.

        Parameters
        ----------
        eval_metric : str (default='r2' for regression, 'accuracy' for classification)
            The evaluation metric to use for model evaluation
        
            * r2, mae, mse, rmse for Regression tasks
            * accuracy, precision, recall, f1_score for Classification tasks
        """
        def highlight_best(s: pd.Series) -> list[str]:
            """
            Highlights the best value in the DataFrame based on the evaluation metric

            Parameters
            ----------
            s : pd.Series
                The Pandas series to apply the highlighting

            Returns
            -------
            list[str]
                A list of strings containing the green background color for the best value so we can highlight it while showing the model stats
            """
            if s.name in ['mae', 'mse', 'rmse']:
                is_best = s == s.min()
            else:
                is_best = s == s.max()
            return ['background-color: green' if v else '' for v in is_best]
        
        eval_metric = self.__eval_metric_checker(eval_metric)
        sorted_model_stats_df = self.__sort_models(eval_metric)
        sorted_model_stats_df['Time Taken (sec)'] = sorted_model_stats_df['Time Taken (sec)'].apply(lambda x: round(x, 2))
        sorted_model_stats_df.index += 1
        
        # If the user is not on a interactive kernel such as Jupyter Notebook, the styled DataFrame will not be displayed
        # Instead, the user will see the raw DataFrame
        # REASON: The styled DataFrame is only supported in interactive kernels, otherwise raises an error
        if get_ipython().__class__.__name__ != 'ZMQInteractiveShell':
            print(20*'-')
            print(sorted_model_stats_df.head(len(self.__ML_MODELS)))
            print(20*'-')

        else:
            # Apply the highlighting to all metric columns and display the dataframe
            styler = sorted_model_stats_df.style.apply(highlight_best, subset=self.__ALL_EVALUATION_METRICS)
            display(styler) # display is only supported in interactive kernels such as Jupyter Notebook, for details check the comment block a couple of lines above

    def plot_feature_importance(self, model: Optional[object] = None):
        """
        Display feature importance for a given model

        Parameters
        ----------
        model: object (default = None)
            Machine learning model to display it's feature importance. If It's set to None, the best model found in the experiment will be used
        """
        try:
            if model is None:
                model = self.get_best_models()

            model_name = model.__class__.__name__
            importance = None

            # Check if the model has 'feature_importances_' attribute (tree-based models)
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_

            # Check if the model has coefficients (linear models)
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)

            if importance is not None:
                indices = np.argsort(importance)[::-1]
                sorted_importance = importance[indices]
                sorted_features = np.array(self.feature_names)[indices]

                plt.figure(figsize=(10, 6))
                plt.barh(range(len(sorted_importance)), sorted_importance, color=plt.cm.viridis(np.linspace(0, 1, len(sorted_importance))))
                plt.yticks(range(len(sorted_features)), sorted_features)
                plt.xlabel("Importance")
                plt.ylabel("Features")
                plt.title(f"Feature Importance for {model_name}")
                plt.gca().invert_yaxis()
                plt.show()

            else:
                self.__logger.info(f"Feature importance is not available for this model {(model_name)}, If you think there is a mistake, please open an issue on GitHub repository")

        except Exception as e:
            self.__logger.error(f"Could not calculate feature importance for the following model: {model}, Error: {e}")

    def tune_model(self, 
                   model: Optional[object] = None,
                   tuning_method: Optional[str] = 'randomized_search',
                   eval_metric: Optional[str] = None,
                   param_grid: Optional[dict] = None,
                   n_iter: int = 10,
                   cv: int = 3,
                   n_jobs: int = -1,
                   verbose: int = 0):
        """
        Tunes the model based on the given parameter grid and evaluation metric.

        Parameters
        ----------
        model : object (default = None)
            The machine learning model to tune. If It's none, flexml retrieves the best model found in the experiment
        
        tuning_method: str (default = 'random_search')
            The tuning method to use for model tuning

            * 'grid_search' for GridSearchCV (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
                Note that GridSearch optimization may take too long to finish since It tries all the possible combinations of the parameters

            * 'randomized_search' for RandomizedSearchCV (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
            
            * 'optuna' for Optuna (https://optuna.readthedocs.io/en/stable/)

        eval_metric : str (default='r2' for regression, 'accuracy' for classification)
            The evaluation metric to use for model evaluation
        
            * r2, mae, mse, rmse for Regression tasks

            * accuracy, precision, recall, f1_score for Classification tasks

        param_grid : dict (default = defined custom param dict in flexml/config/tune_model_config.py)
            The parameter set to use for model tuning.

            Example param_grid for XGBoost
            
            >>>      {
            >>>       'n_estimators': [100, 200, 300],
            >>>       'max_depth': [3, 4, 5],
            >>>       'learning_rate': [0.01, 0.05, 0.1]
            >>>      }

        n_iter : int (default = 10)
            The number of trials to run in the tuning process (Only for RandomizedSearchCV and Optuna)
            
        cv : int (default = 3)
            The number of cross-validation folds to use for the tuning process (Only for GridSearchCV and RandomizedSearchCV)
        
        n_jobs: int (default = -1)
            The number of jobs to run in parallel for the tuning process. -1 means using all threads in the CPU

        verbose: int (default = 0)
            The verbosity level of the tuning process. If It's set to 0, no logs will be shown during the tuning process. Otherwise, the logs will be shown based on the value of the verbose parameter

            * For GridsearchCV and RandomizedSearchCV, the information will be as below:
            
                * >1 : the computation time for each fold and parameter candidate is displayed

                * >2 : the score is also displayed

                * >3 : the fold and candidate parameter indexes are also displayed together with the starting time of the computation

            * For optuna:

                * >DEBUG (Equals to 4): Most detailed logging (prints almost everything)

                * >INFO (Equals to 3): Standard informational output

                * >WARNING (Equals to 2): Only warnings and errors

                * >ERROR (Equals to 1): Only error messages

                * >CRITICAL (Equals to 0): Only critical errors

        Returns
        -------
        tuned_model: object
            The tuned model object
        """
        def _show_tuning_report(tuning_report: dict):
            """
            Shows the tuning report of the model tuning process
            """
            self.tuned_model = tuning_report['tuned_model']
            self.tuned_model_score = tuning_report['tuned_model_score']
            tuned_time_taken = tuning_report['time_taken_sec']
            tuned_model_name = f"{self.tuned_model.__class__.__name__}_({tuning_report['tuning_method']})_(cv={tuning_report['cv']})_(n_iter={tuning_report['n_iter']})"

            # Add the tuned model and it's score to the model_training_info list
            model_perf = self.__evaluate_model_perf(self.y_test, self.tuned_model.predict(self.X_test))
            self.__model_training_info.append({
                tuned_model_name:{
                    "model": self.tuned_model,
                    "model_stats": {
                        "model_name": tuned_model_name,
                        **model_perf,
                        "Time Taken (sec)": tuned_time_taken
                    }
                }
            })
            self.get_best_models() # Update the self.__model_stats_df
            self.show_model_stats()

        eval_metric = self.__eval_metric_checker(eval_metric)

        # Create the ModelTuner object If It's not created before, avoid creating it everytime tune_model() function is called
        if not hasattr(self, 'model_tuner'):
            self.model_tuner = ModelTuner(self.__ML_TASK_TYPE, self.X_train, self.X_test, self.y_train, self.y_test, self.logging_to_file)

        # Get the best model If the user doesn't pass any model object
        if model is None:
            model = self.get_best_models()

        # Get the model's param_grid from the config file If It's not passed from the user
        if param_grid is None:
            try:
                param_grid = [ml_model for ml_model in self.__ML_MODELS if ml_model['name'] == model.__class__.__name__][0]['tuning_param_grid']
                
            except AttributeError:
                error_msg = f"{model}'s tuning config is not found in the config, please pass it manually via 'param_grid' parameter"
                self.__logger.error(error_msg)
                raise ValueError(error_msg)
        
        if not isinstance(n_iter, int) or n_iter < 1:
            info_msg = f"n_iter parameter should be minimum 1, got {n_iter}\nChanged it to 10 for the tuning process"
            n_iter = 10
            self.__logger.info(info_msg)

        if not isinstance(cv, int) or cv < 2:
            info_msg = f"cv parameter should be minimum 2, got {cv}\nChanged it to 2 for the tuning process"
            cv = 2
            self.__logger.info(info_msg)

        if not isinstance(n_jobs, int) or n_jobs < -1:
            info_msg = f"n_jobs parameter should be minimum -1, got {n_jobs}\nChanged it to -1 for the tuning process"
            n_jobs = -1
            self.__logger.info(info_msg)

        self.__logger.info(f"[PROCESS] Model Tuning process is started with '{tuning_method}' method")
        tuning_method = tuning_method.lower()
        if tuning_method == "grid_search":
            tuning_result = self.model_tuner.grid_search(
                model=model,
                param_grid=param_grid,
                eval_metric=eval_metric,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose
            )
            _show_tuning_report(tuning_result)
            
        elif tuning_method == "randomized_search":
            tuning_result = self.model_tuner.random_search(
                model=model,
                param_grid=param_grid,
                eval_metric=eval_metric,
                n_iter=n_iter,
                cv=cv,
                n_jobs=n_jobs,
                verbose=verbose
            )
            _show_tuning_report(tuning_result)
                
        elif tuning_method == "optuna":
            tuning_result = self.model_tuner.optuna_search(
                model=model,
                param_grid=param_grid,
                eval_metric=eval_metric,
                n_iter=n_iter,
                n_jobs=n_jobs,
                verbose=verbose
            )
            _show_tuning_report(tuning_result)
            
        else:
            error_msg = f"Unsupported tuning method: {tuning_method}, expected one of the following: 'grid_search', 'randomized_search', 'optuna'"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
            
        self.__logger.info("[PROCESS] Model Tuning process is finished")