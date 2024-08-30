from typing import Union, Optional
from tqdm import tqdm
import pandas as pd
import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

    experiment_size : str, (default='quick')
        The size of the experiment to run. It can be 'quick' or 'wide'
        
        * If It's selected 'quick', less amount of machine learning models will be used to get quick results
        
        * If It's selected 'wide', wide range of machine learning models will be used to get more comprehensive results

        You can take a look at the models in the library at config/ml_models.py

    test_size : float, (default=0.25)
        The size of the test data in the train-test split process.
    
    random_state : int, (default=42)
        The random state value for the train-test split process
        
        For more info, visit https://scikit-learn.org/stable/glossary.html#term-random_state

    logging_to_file: bool, (default=False)
        If True, the logs will be saved to a file in the current path, located in /logs/flexml_logs.log, Otherwise, it will not be saved.
    """
    def __init__(self,
                 data: pd.DataFrame,
                 target_col: str,
                 experiment_size: str = 'quick',
                 test_size: float = 0.25,
                 random_state: int = 42,
                 logging_to_file: str = False):
        self.data = data
        self.experiment_size = experiment_size
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.logging_to_file = logging_to_file
        self.ML_MODELS = None
        self.__ML_TASK_TYPE = "Regression" if "Regression" in self.__class__.__name__ else "Classification"
        self.__DEFAULT_EVALUATION_METRIC = EVALUATION_METRICS[self.__ML_TASK_TYPE]["DEFAULT"]
        self.__ALL_EVALUATION_METRICS = EVALUATION_METRICS[self.__ML_TASK_TYPE]["ALL"]

        # Logger to log app activities (Logs are stored in flexml/logs/log.log file)
        self.logger = get_logger(__name__, "PROD", self.logging_to_file)

        # Data and ML model preparation stage
        self.__validate_data()
        self.__prepare_models()
        self.__train_test_split()
        self.feature_names = self.data.drop(columns=[self.target_col]).columns
        self.model_training_info = []
        self.model_stats_df = None
        self.eval_metric = None

        # Model Tuning Helper
        self.model_tuner = ModelTuner(self.__ML_TASK_TYPE, self.X_train, self.X_test, self.y_train, self.y_test, self.logging_to_file)

    def __validate_data(self):
        """
        Validates the input data given while initializing the Regression Class
        """
        # Data Overview validation
        if not isinstance(self.data, pd.DataFrame):
            error_msg = f"Dataframe should be a pandas DataFrame, but you've passed {type(self.data)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.data.select_dtypes(include=[np.number]).shape[1] != self.data.shape[1]:
            error_msg = "Dataframe should include only numeric values, did you forget to encode the categorical variables?"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.data.shape[0] == 0:
            error_msg = "Dataframe should include at least one row, is your dataframe empty?"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.data.shape[1] <= 1:
            error_msg = "Dataframe should include at least two columns, consider taking a look at your dataframe"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Target Column validation
        if self.target_col not in self.data.columns:
            error_msg = f"Target column '{self.target_col}' is not found in the data"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.data[self.target_col].isnull().sum() > 0:
            error_msg = "Target column should not include null values"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
    def __prepare_models(self):
        """
        Prepares the models based on the selected experiment size ('quick' or 'wide')
        """
        if not isinstance(self.experiment_size, str):
            error_msg = f"experiment_size expected to be a string, got {type(self.experiment_size)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if self.experiment_size not in ['quick', 'wide']:
            error_msg = f"experiment_size expected to be either 'quick' or 'wide', got {self.experiment_size}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.ML_MODELS = ML_MODELS.get(self.__ML_TASK_TYPE).get(self.experiment_size.upper())
        
    def __train_test_split(self) -> list[np.ndarray]:
        """
        Splits the data into train and test.
        Uses scikit-learn's train_test_split function, for more information, visit https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

        Returns
        -------
        list[np.ndarray]
            A list of arrays containing the train and test data.
        """
        try:   
            X = self.data.drop(columns=[self.target_col])
            y = self.data[self.target_col]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        
        except Exception as e:
            error_msg = f"An error occurred while splitting the data into train and test: {str(e)}"
            self.logger.error(error_msg)
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
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if eval_metric not in self.__ALL_EVALUATION_METRICS:
            error_msg = f"{eval_metric} is not a valid evaluation metric for {self.__ML_TASK_TYPE}, expected one of the following: {self.__ALL_EVALUATION_METRICS}"
            self.logger.error(error_msg)
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
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if top_n_models < 1 or top_n_models > len(self.ML_MODELS):
            error_msg = f"Invalid top_n_models value. Expected a value between 1 and {len(self.ML_MODELS)}, got {top_n_models}"
            self.logger.error(error_msg)
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
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    def start_experiment(self, eval_metric: Optional[str] = None):
        """
        Trains machine learning algorithms and evaluates them based on the specified evaluation metric
        
        Parameters
        ----------
        eval_metric : str (default='r2' for Regression, 'accuracy' for Classification)
            The evaluation metric to use for model evaluation.
        """
        
        self.eval_metric = self.__eval_metric_checker(eval_metric)
        self.model_training_info = [] # Reset the model training info before starting the experiment
        self.model_stats_df = None    # Reset the model stats DataFrame before starting the experiment

        self.logger.info("[PROCESS] Training the ML models")

        for model_idx in tqdm(range(len(self.ML_MODELS))):
            model_info = self.ML_MODELS[model_idx]
            model_name = model_info['name']
            model = model_info['model']
            try:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                model_perf = self.__evaluate_model_perf(self.y_test, y_pred)

                self.model_training_info.append({
                    model_name: {
                        "model": model,
                        "model_stats": {
                            "model_name": model_name,
                            **model_perf}
                    }
                })

            except Exception as e:
                self.logger.error(f"An error occured while training {model_name}: {str(e)}")

        self.logger.info("[PROCESS] Model training is finished!")
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

        for model_info in self.model_training_info:
            if model_name in model_info.keys():
                return model_info[model_name]["model"]
        
        error_msg = f"{model_name} is not found in the trained models, expected one of the following:\n{[list(model_info.keys())[0] for model_info in self.model_training_info]}"
        self.logger.error(error_msg)
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
        if len(self.model_training_info) == 0:
            error_msg = "There is no model performance data to sort!"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        top_n_models = self.__top_n_models_checker(top_n_models)

        if eval_metric is not None:
            eval_metric = self.__eval_metric_checker(eval_metric)
        else: # If the user doesn't pass a eval_metric, get the evaluation metric passed to the start_experiment function
            eval_metric = self.eval_metric
        
        model_stats = []
        best_models = []
        
        for model_pack in self.model_training_info:
            for model_name, model_data in model_pack.items():
                model_stats.append(model_data["model_stats"])
    
        self.model_stats_df = pd.DataFrame(model_stats)
        self.sorted_model_stats_df = self.__sort_models(eval_metric)

        for i in range(top_n_models):
            searched_model_name = self.sorted_model_stats_df.iloc[i]["model_name"]
            for model_info in self.model_training_info:
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
        if len(self.model_stats_df) == 0:
            error_msg = "There is no model performance data to sort!"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        eval_metric = self.__eval_metric_checker(eval_metric)
        
        # Since lower is better for mae, mse and rmse in Regression tasks, they should be sorted in ascending order
        if self.__ML_TASK_TYPE == "Regression" and eval_metric in ['mae', 'mse', 'rmse']:
            return self.model_stats_df.sort_values(by=eval_metric, ascending=True).reset_index(drop = True)
        else:
            return self.model_stats_df.sort_values(by=eval_metric, ascending=False).reset_index(drop = True)

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
        
        # If the user is not on a interactive kernel such as Jupyter Notebook, the styled DataFrame will not be displayed
        # Instead, the user will see the raw DataFrame
        # REASON: The styled DataFrame is only supported in interactive kernels, otherwise raises an error
        if get_ipython().__class__.__name__ != 'ZMQInteractiveShell':
            print(20*'-')
            print(sorted_model_stats_df.head(len(self.ML_MODELS)))
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
                self.logger.info("Feature importance is not available for this model, If you think there is a mistake, please open an issue on GitHub repository")

        except Exception as e:
            self.logger.error(f"Could not calculate feature importance for the following model: {model}, Error: {e}")

    def tune_model(self, 
                   model: Optional[object] = None,
                   tuning_method: Optional[str] = 'randomized_search',
                   tuning_size: Optional[str] = 'wide',
                   eval_metric: Optional[str] = None,
                   param_grid: Optional[dict] = None,
                   n_iter: int = 10,
                   cv: Optional[int] = None,
                   n_jobs: int = -1):
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

        tuning_size: str (default = 'wide')
            The size of the tuning process. It can be 'quick' or 'wide'

            * If 'quick' is selected, number of params or number of values in the each params will be decrased.
                -> For detailed information, visit flexml/_model_tuner.py/_param_grid_validator() function's doc

            * If 'wide' is selected, param_grid will stay same

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
            
        cv : int (default = None)
            The number of cross-validation folds to use for the tuning process (Only for GridSearchCV and RandomizedSearchCV)
        
        n_jobs: int
            The number of jobs to run in parallel for the tuning process. -1 means using all threads in the CPU

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
            tuned_model_name = f"{self.tuned_model.__class__.__name__}_({tuning_report['tuning_method']}({tuning_size}))_(cv={tuning_report['cv']})_(n_iter={tuning_report['n_iter']})"

            # Add the tuned model and it's score to the model_training_info list
            model_perf = self.__evaluate_model_perf(self.y_test, self.tuned_model.predict(self.X_test))
            self.model_training_info.append({
                tuned_model_name:{
                    "model": self.tuned_model,
                    "model_stats": {
                        "model_name": tuned_model_name,
                        **model_perf
                    }
                }
            })
            self.get_best_models() # Update the self.model_stats_df
            self.show_model_stats()

        eval_metric = self.__eval_metric_checker(eval_metric)

        # Get the best model If the user doesn't pass any model object
        if model is None:
            model = self.get_best_models()

        # Get the model's param_grid from the config file If It's not passed from the user
        if param_grid is None:
            try:
                param_grid = [ml_model for ml_model in self.ML_MODELS if ml_model['name'] == model.__class__.__name__][0]['tuning_param_grid']
                
            except AttributeError:
                error_msg = f"{model}'s tuning config is not found in the config, please pass it manually via 'param_grid' parameter"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        
        if cv != None and (not isinstance(cv, int) or cv < 2):
            info_msg = f"cv parameter should be minimum 2, got {cv}\nChanged it to 2 for the tuning process"
            cv = 2
            self.logger.info(info_msg)

        self.logger.info("[PROCESS] Model Tuning process is started")
        tuning_method = tuning_method.lower()
        if tuning_method == "grid_search":
            tuning_result = self.model_tuner.grid_search(
                model=model,
                tuning_size=tuning_size,
                param_grid=param_grid,
                eval_metric=eval_metric,
                cv=cv,
                n_jobs=n_jobs
            )
            _show_tuning_report(tuning_result)
            
        elif tuning_method == "randomized_search":
            tuning_result = self.model_tuner.random_search(
                model=model,
                tuning_size=tuning_size,
                param_grid=param_grid,
                eval_metric=eval_metric,
                n_iter=n_iter,
                cv=cv,
                n_jobs=n_jobs
            )
            _show_tuning_report(tuning_result)
                
        elif tuning_method == "optuna":
            tuning_result = self.model_tuner.optuna_search(
                model=model,
                tuning_size=tuning_size,
                param_grid=param_grid,
                eval_metric=eval_metric,
                n_iter=n_iter,
                n_jobs=n_jobs
            )
            _show_tuning_report(tuning_result)
            
        else:
            error_msg = f"Unsupported tuning method: {tuning_method}, expected one of the following: 'grid_search', 'randomized_search', 'optuna'"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        self.logger.info("[PROCESS] Model Tuning process is finished")