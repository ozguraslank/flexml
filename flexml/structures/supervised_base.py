from typing import Union, Optional
from tqdm import tqdm
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from flexml.config.supervised_config import ML_MODELS, EVALUATION_METRICS
from flexml.logger.logger import get_logger


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
        
        * If It's selected 'quick', quick amount of machine learning models will be used to get quick results
        
        * If It's selected 'wide', wide range of machine learning models will be used to get more comprehensive results
    
    test_size : float, (default=0.25)
        The size of the test data in the train-test split process.
    
    random_state : int, (default=42)
        The random state value for the train-test split process
        
        For more info, visit https://scikit-learn.org/stable/glossary.html#term-random_state

    logging_to_file: bool, (default=True)
        If True, the logs will be saved to a file in the current path, located in /logs/flexml_logs.log, Otherwise, it will not be saved.
    """
    def __init__(self,
                 data: pd.DataFrame,
                 target_col: str,
                 experiment_size: str = 'quick',
                 test_size: float = 0.25,
                 random_state: int = 42,
                 logging_to_file: str = True):
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
        self.logger = get_logger(__name__, self.logging_to_file)

        # Data and ML model preparation stage
        self.__validate_data()
        self.__prepare_models()
        self.__train_test_split()

    def __validate_data(self):
        """
        Validates the input data given while initializing the Regression Class
        """
        # Data Overview validation
        if not isinstance(self.data, pd.DataFrame):
            error_msg = "Dataframe should be a pandas DataFrame"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.data.select_dtypes(include=[np.number]).shape[1] != self.data.shape[1]:
            error_msg = "Dataframe should include only numeric values"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.data.shape[0] == 0:
            error_msg = "Dataframe should include at least one row"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if self.data.shape[1] <= 1:
            error_msg = "Dataframe should include at least two columns"
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
    
    def start_experiment(self,
                     eval_metric: Optional[str] = None,
                     top_n_models: int = 1):
        """
        Trains machine learning algorithms and evaluates them based on the specified evaluation metric.
        Returns the top n models based on the evaluation metric.
        
        Parameters
        ----------
        eval_metric : str (default='r2' for Regression, 'accuracy' for Classification)
            The evaluation metric to use for model evaluation.
        
        top_n_models : int (default=1)
            The number of top models to select based on the evaluation metric.
        """
        eval_metric = self.__eval_metric_checker(eval_metric)
        top_n_models = self.__top_n_models_checker(top_n_models)

        def __evaluate_model_perf(y_test, y_pred):
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
                error_msg = "Unsupported task type"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        
        self.model_training_info = []
        
        self.logger.info("[PROCESS] Training the ML models")

        for model_idx in tqdm(range(len(self.ML_MODELS))):
            model_info = self.ML_MODELS[model_idx]
            model_name = model_info['name']
            model = model_info['model']
            try:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                model_perf = __evaluate_model_perf(self.y_test, y_pred)

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
        self.get_best_models(eval_metric, top_n_models)
        self.show_model_stats(eval_metric)

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
        eval_metric = self.__eval_metric_checker(eval_metric)
        top_n_models = self.__top_n_models_checker(top_n_models)

        if len(self.model_training_info) == 0:
            error_msg = "There is no model performance data to sort!"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        model_stats = []
        best_models = []
        
        for model_pack in self.model_training_info:
            for model_name, model_data in model_pack.items():
                model_stats.append(model_data["model_stats"])
        
        self.model_stats_df = pd.DataFrame(model_stats)
        sorted_model_stats_df = self.__sort_models(eval_metric)

        best_model_names = sorted_model_stats_df.head(top_n_models)["model_name"].tolist()
        for model_pack in self.model_training_info:
            for model_name, model_data in model_pack.items():
                if model_name in best_model_names:
                    best_models.append(model_data.get('model'))
        
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
        eval_metric = self.__eval_metric_checker(eval_metric)

        if len(self.model_stats_df) == 0:
            error_msg = "There is no model performance data to sort!"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
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
        eval_metric = self.__eval_metric_checker(eval_metric)
        sorted_model_stats_df = self.__sort_models(eval_metric)

        print(sorted_model_stats_df.head(len(self.ML_MODELS)))