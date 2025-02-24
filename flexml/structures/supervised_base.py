import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from time import time
from typing import Any, Union, Optional, Iterator, List, Dict
from tqdm import tqdm
from IPython import get_ipython
from sklearn.pipeline import Pipeline
from flexml.config import ML_MODELS, EVALUATION_METRICS, CROSS_VALIDATION_METHODS
from flexml.logger import get_logger
from flexml.helpers import (
    eval_metric_checker,
    random_state_checker,
    cross_validation_checker,
    get_cv_splits,
    evaluate_model_perf,
    validate_inputs
)
from flexml._model_tuner import ModelTuner
from flexml._feature_engineer import FeatureEngineering

import warnings
warnings.filterwarnings("ignore")


class SupervisedBase:
    """
    Base class for Supervised tasks (regression & classification)

    Parameters
    ----------
    data : pd.DataFrame
        The input data for the model training process
    
    target_col : str
        The target column name in the data

    random_state : int, optional (default=42)
        The random state value for the data processing process (Ignored If 'shuffle' is set to False)

        If None, It uses the global random state instance from numpy.random. Thus, It will produce different results in every execution of start_experiment()

    drop_columns : list, default=None
        Columns that will be dropped from the data
    
    categorical_imputation_method : str, default='mode'
        Imputation method for categorical columns. Options:
        * 'mode': Replace missing values with the most frequent value
        * 'constant': Replace missing values with a constant value
        * 'drop': Drop rows with missing values

    numerical_imputation_method : str, default='mean'
        Imputation method for numerical columns. Options:
        * 'mean': Replace missing values with the column mean
        * 'median': Replace missing values with the column median
        * 'mode': Replace missing values with the column mode
        * 'constant': Replace missing values with a constant value
        * 'drop': Drop rows with missing values

    column_imputation_map : dict, default=None
        Custom mapping of columns to specific imputation methods
        Example usage: {'column_name': 'mean', 'column_name2': 'mode'}

    numerical_imputation_constant : float, default=0.0
        The constant value for imputing numerical columns when 'constant' is selected

    categorical_imputation_constant : str, default='Unknown'
        The constant value for imputing categorical columns when 'constant' is selected

    encoding_method : str, default='onehot_encoder'
        Encoding method for categorical columns. Options:
        * 'label_encoder': Use label encoding
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
        * 'onehot_encoder': Use one-hot encoding
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        * 'ordinal_encoder': Use ordinal encoding
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
        
    onehot_limit : int, default=25
        Maximum number of categories to use for one-hot encoding

    encoding_method_map : dict, default=None
        Custom mapping of columns to encoding methods
        Example usage: {'column_name': 'onehot_encoder', 'column_name2': 'label_encoder'}
    
    ordinal_encode_map : dict, default=None
        Custom mapping of columns to category order for ordinal encoding
        Example usage: {'column_name': ['low', 'medium', 'high']}
    
    normalize : str, default=None
        Standardize the data using StandardScaler. Options:
        * 'standard_scaler': Standardize the data using StandardScaler
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        * 'minmax_scaler': Scale the data using MinMaxScaler
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
        * 'robust_scaler': Scale the data using RobustScaler
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
        * 'quantile_transformer': Transform the data using QuantileTransformer
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
        * 'maxabs_scaler': Scale the data using MaxAbsScaler
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html
        * 'normalize_scaler': Normalize the data to unit length
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html

    shuffle: bool, (default=True)
        If True, the data will be shuffled before the model training process

    logging_to_file: bool, (default=False)
        If True, the logs will be saved to a file in the current path, located in /logs/flexml_logs.log, Otherwise, it will not be saved
    """
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        random_state: Optional[int] = 42,
        drop_columns: Optional[List[str]] = None,
        categorical_imputation_method: str = "mode",
        numerical_imputation_method: str = "mean", 
        column_imputation_map: Optional[Dict[str, str]] = None,
        numerical_imputation_constant: float = 0.0,
        categorical_imputation_constant: str = "Unknown", 
        encoding_method: str = "onehot_encoder",
        onehot_limit: int = 25,
        encoding_method_map: Optional[Dict[str, str]] = None,
        ordinal_encode_map: Optional[Dict[str, List[str]]] = None,
        normalize: Optional[str] = None,
        shuffle: bool = True,
        logging_to_file: str = False,
    ):

        # Logger to log app activities (Logs are stored in /logs/flexml_logs.log file if logging_to_file is passed as True)
        self.__logger = get_logger(__name__, "PROD", logging_to_file)

        random_state = random_state_checker(random_state)
        self._data_processing_random_state = random_state
        
        self.data = data
        self.target_col = target_col
        self.logging_to_file = logging_to_file
        self.shuffle = shuffle

        self.feature_engineering_params = {
            'data': data,
            'target_col': target_col,
            'drop_columns': drop_columns,
            'categorical_imputation_method': categorical_imputation_method,
            'numerical_imputation_method': numerical_imputation_method,
            'column_imputation_map': column_imputation_map,
            'numerical_imputation_constant': numerical_imputation_constant,
            'categorical_imputation_constant': categorical_imputation_constant,
            'encoding_method': encoding_method,
            'onehot_limit': onehot_limit,
            'encoding_method_map': encoding_method_map,
            'ordinal_encode_map': ordinal_encode_map,
            'normalize': normalize,
        }

        # Data Preparation
        self.__validate_data()
        validate_inputs(**self.feature_engineering_params)
        self.X = self.data.drop(columns=[self.target_col])
        self.y = self.data[self.target_col] 

        self.drop_columns = drop_columns
        self.categorical_imputation_method = categorical_imputation_method
        self.numerical_imputation_method = numerical_imputation_method
        self.column_imputation_map = column_imputation_map
        self.numerical_imputation_constant = numerical_imputation_constant
        self.categorical_imputation_constant = categorical_imputation_constant
        self.encoding_method = encoding_method
        self.onehot_limit = onehot_limit
        self.encoding_method_map = encoding_method_map
        self.ordinal_encode_map = ordinal_encode_map
        self.normalize = normalize
        self.feature_names = self.data.drop(columns=[self.target_col]).columns
        self.full_data_feature_engineer = None

        # Model Preparation
        self.__ML_MODELS = []
        self.__ML_TASK_TYPE = "Regression" if "Regression" in self.__class__.__name__ else "Classification"
        self.__ALL_EVALUATION_METRICS = EVALUATION_METRICS[self.__ML_TASK_TYPE]["ALL"]
        self.__existing_model_names = [] # To keep the existing model names in the experiment
        self.__models_raised_error = []   # To keep the models that raised error in the experiment to avoid running them again in the next cv splits
        self.__model_training_info = []
        self.__model_stats_df = None
        self.__sorted_model_stats_df = None

        # Cross-Validation Settings
        self.__AVAILABLE_CV_METHODS = CROSS_VALIDATION_METHODS[self.__ML_TASK_TYPE]

        # Keep the start_experiment params in memory to avoid re-creating cv_splits again for no-data-change conditions in start_experiment and tune_model
        self._last_training_random_state = None
        self._last_cv_method = None
        self._last_n_folds = None
        self._last_test_size = None
        self._last_groups_col = None
        self._last_experiment_size = None

        # Track experiment history
        self._experiment_history = []

    def __repr__(self):
        return (
            f"SupervisedBase(\n"
            f"data={self.data.head()},\n"
            f"target_col={self.target_col},\n"
            f"random_state={self._data_processing_random_state},\n"
            f"drop_columns={self.drop_columns},\n"
            f"categorical_imputation_method={self.categorical_imputation_method},\n"
            f"numerical_imputation_method={self.numerical_imputation_method},\n"
            f"column_imputation_map={self.column_imputation_map},\n"
            f"numerical_imputation_constant={self.numerical_imputation_constant},\n"
            f"categorical_imputation_constant={self.categorical_imputation_constant},\n"
            f"encoding_method={self.encoding_method},\n"
            f"onehot_limit={self.onehot_limit},\n"
            f"encoding_method_map={self.encoding_method_map},\n"
            f"ordinal_encode_map={self.ordinal_encode_map},\n"
            f"normalize={self.normalize},\n"
            f"shuffle={self.shuffle},\n"
            f"logging_to_file={self.logging_to_file}\n"
            f")"
        )
    
    def __validate_data(self):
        """
        Validates the input data given while initializing the Regression Class
        """
        # Data Overview validation
        if not isinstance(self.data, pd.DataFrame):
            error_msg = f"Dataframe should be a pandas DataFrame, but you've passed {type(self.data)}"
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
        
    def __prepare_models(self, experiment_size: str):
        """
        Prepares the models based on the selected experiment size ('quick' or 'wide')

        Parameters
        ----------
        experiment_size : str
            The size of the experiment to run. It can be 'quick' or 'wide'
        """
        if not isinstance(experiment_size, str):
            error_msg = f"experiment_size expected to be a string, got {type(experiment_size)}"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)

        if experiment_size not in ['quick', 'wide']:
            error_msg = f"experiment_size expected to be either 'quick' or 'wide', got {experiment_size}"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.__ML_MODELS = ML_MODELS.get(self.__ML_TASK_TYPE).get(experiment_size.upper())
    
    def __top_n_models_checker(self, top_n_models: Optional[int]) -> int:
        """
        Validates the top_n_models parameter taken from the user

        Parameters
        ----------
        top_n_models : int
            The number of top models to select based on the evaluation metric
        """
        if top_n_models is None:
            return 1
        
        if top_n_models < 1 or top_n_models > len(self.__ML_MODELS):
            error_msg = f"Invalid top_n_models value. Expected a value between 1 and {len(self.__ML_MODELS)}, got {top_n_models}"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        return top_n_models
    
    def __process_experiment_result(self, experiment_stats: dict):
        """
        Processes and aggregates the results of an experiment, calculating average metrics and selecting the best model.

        Parameters
        ----------
        experiment_stats : dict
            A dictionary containing experiment results. The keys are model names, and the values are lists of model entries, 
            where each entry is a dictionary containing "model_stats" (a dictionary of metrics) and the trained "model"
        """
        for model_name, model_entries in experiment_stats.items():
            # Aggregate metrics across all entries for this model
            aggregated_metrics = defaultdict(list)

            for entry in model_entries:
                for key, value in entry["model_stats"].items():
                    aggregated_metrics[key].append(value)
            
            # Calculate the average for all aggregated metrics, with a special case for "Time Taken (sec)"
            averaged_metrics = {
                key: np.sum(value) if key == "Time Taken (sec)" else np.mean(value) if isinstance(value[0], (int, float)) else value[0]
                for key, value in aggregated_metrics.items()
            }
            
            best_model_entry = max(model_entries, key=lambda x: x["model_stats"][self.eval_metric])
            
            self.__model_training_info.append({
                model_name: {
                    "model": best_model_entry["model"],  # Use the best model based on the max metric value
                    "model_stats": averaged_metrics
                }
            })
    
    def start_experiment(
        self,
        experiment_size: str = 'quick',
        cv_method: Optional[str] = None,
        n_folds: Optional[int] = None,
        test_size: Optional[float] = None,
        eval_metric: Optional[str] = None,
        random_state: Optional[int] = 42,
        groups_col: Optional[str] = None
    ):
        """
        Trains machine learning algorithms and evaluates them based on the specified evaluation metric

        Parameters
        ----------
        experiment_size : str, (default='quick')
            The size of the experiment to run. It can be 'quick' or 'wide'
            - 'quick': Uses fewer models for faster results.
            - 'wide': Uses more models for comprehensive results.
            Models are defined in config/ml_models.py

        cv_method : str, (default='kfold' for Regression, 'stratified_kfold' for Classification)
            Cross-validation method to use. Options:

            - For Regression:
                - "kfold" (default) (Provide `n_folds`)
                - "holdout" (Provide `test_size`)
                - "shuffle_split" (Provide `n_folds` and `test_size`)
                - "group_kfold" (Provide `n_folds` and `groups_col`)
                - "group_shuffle_split" (Provide `n_folds`, `test_size`, and `groups_col`)
            
            - For Classification:
                - "kfold" (default) (Provide `n_folds`)
                - "stratified_kfold" (default) (Provide `n_folds`)
                - "holdout" (Provide `test_size`)
                - "stratified_shuffle_split" (Provide `n_folds`, `test_size`)
                - "group_kfold" (Provide `n_folds` and `groups_col`)
                - "group_shuffles_plit" (Provide `n_folds`, `test_size`, and `groups_col`)

        n_folds : int, optional (default=None for hold-out validation, 5 for other cv methods)
            Number of folds for cross-validation methods

        test_size : float, (default=0.25 for hold-out cv, None for other methods)
            The size of the test data if using hold-out or shuffle-based splits

        eval_metric : str (default='R2' for Regression, 'Accuracy' for Classification)
            The evaluation metric to use for model evaluation

        random_state : int, optional (default=42)
            The random state value for the model training process
            # TODO: Not implemented yet, will be implemented in 1.1.0 release

        groups_col : str, optional
            Column name for group-based cross-validation methods

        Notes for Cross-Validation Methods
        ----------------------------------
        - Group-based methods require `groups_col` to define group labels
        - If both `n_folds` and `test_size` are provided, shuffle-based methods are prioritized
        - Defaults to a standard 5-fold if neither `n_folds` nor `test_size` is provided
        """
        self.eval_metric = eval_metric_checker(self.__ML_TASK_TYPE, eval_metric)
        random_state = random_state_checker(random_state)

        # Check cross-validation method params
        cv_method = cross_validation_checker(
            df=self.data,
            cv_method=cv_method,
            n_folds=n_folds,
            test_size=test_size,
            groups_col=groups_col,
            available_cv_methods=self.__AVAILABLE_CV_METHODS,
            ml_task_type=self.__ML_TASK_TYPE
        )
        if cv_method != "holdout":
            n_folds = 5

        # Check if the cross-validation parameters are changed or not, If they are changed, re-create the cv_splits
        params_changed_flag = (
            self._last_cv_method != cv_method
            or self._last_n_folds != n_folds
            or self._last_test_size != test_size
            or self._last_groups_col != groups_col
        )

        shuffle_with_no_random_state_flag = self.shuffle and self._data_processing_random_state is None
        # If the user selected random_state as None in class definition, cv_splits should be re-created in each experiment --
        # since None value for random_state means different random seed in each execution

        reset_the_experiment = params_changed_flag or shuffle_with_no_random_state_flag
        quick_to_wide_flag = self._last_experiment_size == 'quick' and experiment_size == 'wide'
        
        # Set the current experiment size
        self._last_experiment_size = experiment_size

        # if any cross-validation related parameter is changed, re-create the cv_splits
        if reset_the_experiment:
            self._last_cv_method = cv_method
            self._last_n_folds = n_folds
            self._last_test_size = test_size
            self._last_groups_col = groups_col

            self.cv_splits = list(get_cv_splits(
                df=self.data,
                cv_method=cv_method,
                n_folds=n_folds,
                test_size=test_size,
                groups_col=groups_col,
                random_state=self._data_processing_random_state,
                shuffle=self.shuffle,
                ml_task_type=self.__ML_TASK_TYPE,
                logging_to_file=self.logging_to_file,
                y_array = self.data[self.target_col]
            ))

        self.__prepare_models(experiment_size)
        cv_splits_copy = self.cv_splits.copy() # Will be used for trainings

        self.__logger.info(f"[PROCESS] Training the ML models with {cv_method} validation")

        # Train only the models that haven't been trained yet
        if not reset_the_experiment and quick_to_wide_flag: 
            self.__existing_model_names = self.__model_stats_df['model_name'].unique() if self.__model_stats_df is not None else []
        else:
            self.__model_training_info = []
            self.__model_stats_df = None
            self.__existing_model_names = []
            self.__models_raised_error = []
        
        all_model_stats = defaultdict(list)
        total_iterations = len(cv_splits_copy) * len(self.__ML_MODELS)

        feature_engineer = FeatureEngineering(**self.feature_engineering_params)
        feature_engineer.setup()
        feature_engineer.check_column_anomalies()

        with tqdm(total=total_iterations, desc="INFO | Training Progress", bar_format="{desc}:  | {bar} | {percentage:.0f}%") as pbar:
            for train_idx, test_idx in cv_splits_copy:
                X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

                feature_engineer.data = pd.concat([X_train, y_train], axis=1)
                feature_engineer.setup()
                X_train = feature_engineer.start_feature_engineering().drop(self.target_col, axis=1)
                X_test = feature_engineer.transform_new_data(X_test)

                for model_idx in range(len(self.__ML_MODELS)):
                    model_info = self.__ML_MODELS[model_idx]
                    model_name = model_info['name']
                    
                    if model_name in self.__existing_model_names or model_name in self.__models_raised_error:
                        pbar.update(1)
                        continue  # Skip already trained or raised error models

                    model = model_info['model']
                    try:
                        all_metrics = []
                        all_times = []

                        t_start = time()
                        model.fit(X_train, y_train)
                        t_end = time()

                        time_taken = round(t_end - t_start, 2)
                        y_pred = model.predict(X_test)
                        model_perf = evaluate_model_perf(self.__ML_TASK_TYPE, y_test, y_pred)

                        all_metrics.append(model_perf)
                        all_times.append(time_taken)

                        # Aggregate metrics across all folds
                        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
                        total_time_taken = np.sum(all_times)

                        # Store results temporarily in a defaultdict to group by model name
                        all_model_stats[model_name].append({
                            "model": model,
                            "model_stats": {
                                "model_name": model_name,
                                **avg_metrics,
                                "Time Taken (sec)": total_time_taken
                            }
                        })

                    except Exception as e:
                        self.__logger.error(f"An error occurred while training {model_name}: {str(e)}")
                        self.__models_raised_error.append(model_name)

                    finally:
                        pbar.update(1)

        self.__process_experiment_result(all_model_stats)

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
        eval_metric : str (default='R2 for Regression, 'Accuracy' for Classification)
            The evaluation metric to use for model evaluation:
                
                * R2, MAE, MSE, RMSE, MAPE for Regression tasks

                * Accuracy, Precision, Recall, F1 Score for Classification tasks
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
            eval_metric = eval_metric_checker(self.__ML_TASK_TYPE, eval_metric)
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
                    if top_n_models == 1: # If top_n_models is 1 and there are more than 1 model with the same name, avoid user to get multiple models by stopping the loop
                        break
        
        if len(best_models) == 1:
            return best_models[0]
        
        return best_models
    
    def save_model(
        self,
        model: Optional[Union[str, object]] = None,
        save_path: Optional[str] = None,
        model_only: bool = False,
        full_train: bool = True
    ):
        """
        Saves a specified model or the best model based on evaluation metrics
        and integrates feature engineering into the pipeline
        
        Parameters
        ----------
        model : object, optional
            The model to save. If None, the best model will be fetched
        save_path : str, optional
            The path to save the pipeline
        include_feature_pipeline : bool, optional
            Whether to include the feature engineering pipeline in the saved pipeline
        full_train : bool, optional
            Whether to train the model using the fully feature-engineered data
            
        Returns
        -------
        Pipeline or object
        """
        # Ensure save_path is defined
        if save_path is None:
            save_path = "pipeline.pkl" if not model_only else "model.pkl"
            self.__logger.info(f"No save path provided. Using default: {save_path}")

        if not save_path.endswith(".pkl"):
            self.__logger.warning(f"Only .pkl files are supported. Changing '{save_path}' to '{save_path.rsplit('.', 1)[0]}.pkl'.")
            save_path = save_path.rsplit('.', 1)[0] + ".pkl"

        # Initialize pipeline steps
        pipeline_steps = []

        # Initialize and setup feature engineering if needed
        if not model_only or full_train:
            if self.full_data_feature_engineer is None:
                self.full_data_feature_engineer = FeatureEngineering(**self.feature_engineering_params)
                self.full_data_feature_engineer.setup()
            
            if not model_only:
                # Add the feature engineering pipeline directly
                pipeline_steps.extend(self.full_data_feature_engineer.pipeline.steps)

        # Fetch the best model if no specific model is provided
        if model is None:
            try:
                model = self.get_best_models()
            except ValueError as e:
                error_msg = "No models have been evaluated yet, and no model was specified to save."
                self.__logger.error(error_msg)
                raise ValueError(error_msg) from e
        elif isinstance(model, str):
            try:
                model = self.get_model_by_name(model)
            except KeyError:
                error_msg = f"Model with name '{model}' not found."
                self.__logger.error(error_msg)
                raise ValueError(error_msg)

        # Handle full training scenario if required
        if full_train:
            transformed_data = self.full_data_feature_engineer.start_feature_engineering()
            self.__logger.info("Training the model using the fully feature-engineered data")
            model.fit(
                transformed_data.drop(columns=[self.target_col]), 
                transformed_data[self.target_col]
            )

        # If no feature pipeline is included, return the model directly
        if model_only:
            try:
                with open(save_path, 'wb') as f:
                    pickle.dump(model, f)
                self.__logger.info(f"Model saved successfully at {save_path}")
            except Exception as e:
                self.__logger.error(f"Failed to save model: {e}")
                raise
            
            return model

        # Add the model to the pipeline
        pipeline_steps.append(('model', model))

        # Create the pipeline
        pipeline = Pipeline(pipeline_steps)

        # Save the pipeline
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(pipeline, f)
            self.__logger.info(f"Pipeline saved successfully at {save_path}")
        except Exception as e:
            self.__logger.error(f"Failed to save pipeline: {e}")
            raise

        return pipeline

    def _predict_helper(
        self,
        test_data: pd.DataFrame,
        model: Optional[Union[str, object]] = None,
        full_train: bool = True
    ) -> tuple:
        """Inner handler for prediction methods that returns prepared model and transformed data"""
        if test_data is None or test_data.empty:
            raise ValueError("test_data must be provided and non-empty")

        # Check column consistency
        expected_columns = set(self.X.columns)
        test_columns = set(test_data.columns)
        if expected_columns != test_columns:
            missing = expected_columns - test_columns
            extra = test_columns - expected_columns
            error_msg = "Mismatch in test_data columns."
            if missing: error_msg += f" Missing: {missing}."
            if extra: error_msg += f" Extra: {extra}."
            raise ValueError(error_msg)

        if model is None:
            model = self.get_best_models()
        elif isinstance(model, str):
            model = self.get_model_by_name(model)

        if self.full_data_feature_engineer is None:
            self.full_data_feature_engineer = FeatureEngineering(**self.feature_engineering_params)
            self.full_data_feature_engineer.setup()
        
        # Prepare training data if needed
        transformed_train_data = self.full_data_feature_engineer.start_feature_engineering()
        if full_train:
            self.__logger.info("Training model with full feature-engineered data")
            model.fit(transformed_train_data.drop(columns=[self.target_col]), 
                     transformed_train_data[self.target_col])

        # Transform test data
        transformed_test = self.full_data_feature_engineer.transform_new_data(test_data)
        return model, transformed_test

    def predict(
        self,
        test_data: pd.DataFrame,
        model: Optional[Union[str, object]] = None,
        full_train: bool = True
    ) -> np.ndarray:
        """
        Predicts the target column using the specified or best model

        Parameters
        ----------
        test_data : pd.DataFrame
            The input data to predict the target column
        model : str or object, optional
            The trained model or model name to fetch for prediction
            If None, the best model will be fetched
        full_train : bool, optional
            Whether to train the model using the fully feature-engineered data before prediction

        Returns
        -------
        np.ndarray
            The predicted target column
        """
        model, transformed_test = self._predict_helper(test_data, model, full_train)
        return model.predict(transformed_test)
    
    def predict_proba(
        self,
        test_data: pd.DataFrame,
        model: Optional[Union[str, object]] = None,
        full_train: bool = True
    ) -> np.ndarray:
        """
        Predicts the target column probabilities using the specified or best model

        Parameters
        ----------
        test_data : pd.DataFrame
            The input data to predict the target column
        model : str or object, optional
            The trained model or model name to fetch for prediction
            If None, the best model will be fetched
        full_train : bool, optional
            Whether to train the model using the fully feature-engineered data before prediction

        Returns
        -------
        np.ndarray
            The predicted probabilities for each class
        """
        model, transformed_test = self._predict_helper(test_data, model, full_train)
        return model.predict_proba(transformed_test)

    def __sort_models(self, eval_metric: Optional[str] = None):
        """
        Sorts the models based on the evaluation metric.

        Parameters
        ----------
        eval_metric : str (default='R2')
            The evaluation metric to use for model evaluation

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the sorted model statistics according to the desired eval_metric
        """
        if len(self.__model_stats_df) == 0:
            error_msg = "There is no model performance data to sort!"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        eval_metric = eval_metric_checker(self.__ML_TASK_TYPE, eval_metric)
        
        # Since lower is better for mae, mse and rmse in Regression tasks, they should be sorted in ascending order
        if self.__ML_TASK_TYPE == "Regression" and eval_metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
            return self.__model_stats_df.sort_values(by=eval_metric, ascending=True).reset_index(drop = True)
        else:
            return self.__model_stats_df.sort_values(by=eval_metric, ascending=False).reset_index(drop = True)

    def show_model_stats(self, eval_metric: Optional[str] = None):
        """
        Sorts and shows the model statistics table based on the evaluation metric.

        Parameters
        ----------
        eval_metric : str (default='R2' for regression, 'Accuracy' for classification)
            The evaluation metric to use for model evaluation
        
            * R2, MAE, MSE, RMSE, MAPE for Regression tasks
            * Accuracy, Precision, Recall, F1 Score for Classification tasks
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
            if s.name in ['MAE', 'MSE', 'RMSE', 'MAPE']:
                s_nonneg = s.where(s >= 0, np.nan)
                best_val = s_nonneg.min()
                is_best = s == best_val
            else:
                is_best = s == s.max()
            return ['background-color: green' if v else '' for v in is_best]
        
        eval_metric = eval_metric_checker(self.__ML_TASK_TYPE, eval_metric)
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
            # Apply the highlighting to all metric columns and display the dataframe If It has more than 1 row so that we can compare the models
            if len(sorted_model_stats_df) < 2:
                display(sorted_model_stats_df)
            
            else:
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

    def tune_model(
        self, 
        model: Optional[object] = None,
        tuning_method: Optional[str] = 'randomized_search',
        n_iter: int = 10,
        cv_method: Optional[str] = None,
        n_folds: Optional[int] = None,
        test_size: Optional[float] = None,
        groups_col: Optional[str] = None,
        eval_metric: Optional[str] = None,
        param_grid: Optional[dict] = None,
        n_jobs: int = -1,
        verbose: int = 0
    ):
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

        n_iter : int (default = 10)
            The number of trials to run in the tuning process (Only for RandomizedSearchCV and Optuna)

        cv_method : str, (default='kfold' for Regression, 'stratified_kfold' for Classification)
            Cross-validation method to use. Options:

            - For Regression:
                - "kfold" (default) (Provide `n_folds`)
                - "holdout" (Provide `test_size`)
                - "shuffle_split" (Provide `n_folds` and `test_size`)
                - "group_kfold" (Provide `n_folds` and `groups_col`)
                - "group_shuffle_split" (Provide `n_folds`, `test_size`, and `groups_col`)
            
            - For Classification:
                - "kfold" (default) (Provide `n_folds`)
                - "stratified_kfold" (default) (Provide `n_folds`)
                - "holdout" (Provide `test_size`)
                - "stratified_shuffle_split" (Provide `n_folds`, `test_size`)
                - "group_kfold" (Provide `n_folds` and `groups_col`)
                - "group_shuffles_plit" (Provide `n_folds`, `test_size`, and `groups_col`)
            
        n_folds : int, optional (default=None for hold-out validation, 5 for other cv methods)
            The number of cross-validation folds to use for the tuning process (Only for GridSearchCV and RandomizedSearchCV)
        
        test_size : float, (default=0.25 for hold-out cv, None for other methods)
            The size of the test data if using hold-out or shuffle-based splits

        groups_col : str, optional
            Column name for group-based cross-validation methods

        eval_metric : str (default='R2' for regression, 'Accuracy' for classification)
            The evaluation metric to use for model evaluation
        
            * R2, MAE, MSE, RMSE, MAPE for Regression tasks

            * Accuracy, Precision, Recall, F1 Score for Classification tasks

        param_grid : dict (default = defined custom param dict in flexml/config/tune_model_config.py)
            The parameter set to use for model tuning.

            Example param_grid for XGBoost
            
            >>>      {
            >>>       'n_estimators': [100, 200, 300],
            >>>       'max_depth': [3, 4, 5],
            >>>       'learning_rate': [0.01, 0.05, 0.1]
            >>>      }

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
        def _show_tuning_report(tuning_report: Optional[dict] = None):
            """
            Shows the tuning report of the model tuning process

            Parameters
            ----------
            tuning_report: dict (default = None)
                The tuning report of the model tuning process
            """
            if tuning_report is None:
                return
            
            self.tuned_model = tuning_report['tuned_model']
            self.tuned_model_score = tuning_report['tuned_model_score']
            tuned_time_taken = tuning_report['time_taken_sec']
            tuned_model_name = f"{self.tuned_model.__class__.__name__}_({tuning_report['tuning_method']})_(n_iter={tuning_report['n_iter']})"

            # Add the tuned model and it's score to the model_training_info list
            model_perf = tuning_report['model_perf']
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

        eval_metric = eval_metric_checker(self.__ML_TASK_TYPE, eval_metric)
        
        # Check cross-validation method params
        cv_method = cross_validation_checker(
            df=self.data,
            cv_method=cv_method,
            n_folds=n_folds,
            test_size=test_size,
            groups_col=groups_col,
            available_cv_methods=self.__AVAILABLE_CV_METHODS,
            ml_task_type=self.__ML_TASK_TYPE
        )
        if cv_method != "holdout":
            n_folds = 5

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
            info_msg = f"n_iter parameter should be minimum 1, got {n_iter}, Changed it to 10 for the tuning process"
            n_iter = 10
            self.__logger.info(info_msg)

        if not isinstance(n_jobs, int) or n_jobs < -1:
            info_msg = f"n_jobs parameter should be minimum -1, got {n_jobs}, Changed it to -1 for the tuning process"
            n_jobs = -1
            self.__logger.info(info_msg)

        # If tune_model cross validation params are same, get the current one
        if hasattr(self, 'cv_splits') and cv_method == self._last_cv_method and n_folds == self._last_n_folds and test_size == self._last_test_size and groups_col == self._last_groups_col:
            cv_obj = self.cv_splits
        else:
            if self.__model_stats_df is not None:
                self.__logger.warning("[WARNING] Validation params you've provided are different than the last run. Model performance table will be erased")
                self.__model_stats_df = None
                self.__model_training_info = []
                self.__existing_model_names = []

                self._last_cv_method = cv_method
                self._last_n_folds = n_folds
                self._last_test_size = test_size
                self._last_groups_col = groups_col

            cv_obj = list(get_cv_splits(
                df=self.data,
                cv_method=cv_method,
                n_folds=n_folds,
                test_size=test_size,
                y_array=self.data[self.target_col], 
                groups_col=groups_col,
                ml_task_type=self.__ML_TASK_TYPE,
                logging_to_file=self.logging_to_file
            ))

        # Create the ModelTuner object If It's not created before, avoid creating it everytime tune_model() function is called
        if not hasattr(self, 'model_tuner'):
            self.model_tuner = ModelTuner(self.__ML_TASK_TYPE, self.X, self.y, self.logging_to_file)

        feature_engineer = FeatureEngineering(**self.feature_engineering_params)
        feature_engineer.setup()
        pipeline = feature_engineer.pipeline
        pipeline = Pipeline(steps=pipeline.steps + [('model', model)])

        self.__logger.info(f"[PROCESS] Model Tuning process started with '{tuning_method}' method")
        tuning_method = tuning_method.lower()
        if tuning_method == "grid_search":
            tuning_result = self.model_tuner.grid_search(
                pipeline=pipeline,
                param_grid=param_grid,
                eval_metric=eval_metric,
                cv=cv_obj,
                n_jobs=n_jobs,
                verbose=verbose
            )
            
        elif tuning_method == "randomized_search":
            tuning_result = self.model_tuner.random_search(
                pipeline=pipeline,
                param_grid=param_grid,
                eval_metric=eval_metric,
                n_iter=n_iter,
                cv=cv_obj,
                n_jobs=n_jobs,
                verbose=verbose
            )
                
        elif tuning_method == "optuna":
            tuning_result = self.model_tuner.optuna_search(
                pipeline=pipeline,
                param_grid=param_grid,
                eval_metric=eval_metric,
                cv=cv_obj,
                n_iter=n_iter,
                n_jobs=n_jobs,
                verbose=verbose
            )
            
        else:
            error_msg = f"Unsupported tuning method: {tuning_method}, expected one of the following: 'grid_search', 'randomized_search', 'optuna'"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        _show_tuning_report(tuning_result)
        self.__logger.info("[PROCESS] Model Tuning process is finished")
