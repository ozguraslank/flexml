import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from copy import deepcopy
from time import time
from typing import Union, Optional, List, Dict
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from sklearn.pipeline import Pipeline
from flexml.logger import get_logger
from flexml.config import (
    get_ml_models,
    EVALUATION_METRICS,
    CROSS_VALIDATION_METHODS,
    PLOT_TYPES
)
from flexml.helpers import (
    eval_metric_checker,
    random_state_checker,
    cross_validation_checker,
    get_cv_splits,
    evaluate_model_perf,
    validate_inputs,
    is_interactive_notebook,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_shap,
    plot_residuals,
    plot_prediction_error,
    plot_calibration_curve
)
from flexml._model_tuner import ModelTuner
from flexml._feature_engineer import FeatureEngineering

import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


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

    categorical_imputation_constant : str, default='Unknown'
        The constant value for imputing categorical columns when 'constant' is selected

    numerical_imputation_constant : float, default=0.0
        The constant value for imputing numerical columns when 'constant' is selected

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
        categorical_imputation_constant: str = "Unknown", 
        numerical_imputation_constant: float = 0.0,
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
            'categorical_imputation_constant': categorical_imputation_constant,
            'numerical_imputation_constant': numerical_imputation_constant,
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
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.num_class = len(self.y.unique())
        self.feature_engineer = FeatureEngineering(**self.feature_engineering_params)
        self.feature_engineer.setup()
        self.feature_engineer.check_column_anomalies()

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

        # Model Preparation
        self.__ML_MODELS = []
        self.__ML_TASK_TYPE = "Regression" if "Regression" in self.__class__.__name__ else "Classification"
        self.__ALL_EVALUATION_METRICS = EVALUATION_METRICS[self.__ML_TASK_TYPE]["ALL"]
        self.__existing_model_names = [] # To keep the existing model names in the experiment
        self.__models_raised_error = []   # To keep the models that raised error in the experiment to avoid running them again in the next cv splits
        self.__model_training_info = []
        self._model_stats_df = None
        self._holdout_model_objects = {}
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

        self.__repr__()

    def __repr__(self):
        any_imputation_method_constant_flag = self.categorical_imputation_method == "constant" or self.numerical_imputation_method == "constant"
        width = 30 if any_imputation_method_constant_flag else 25

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Attribute", style="dim", width=width)
        table.add_column("Value")

        # Add regular fields to the table
        table.add_row("Data Shape", str(self.data.shape))
        table.add_row("Target Column", self.target_col)
        table.add_row("Random State", str(self._data_processing_random_state))

        # Only add "Drop Columns" if it's not empty or None
        if self.drop_columns:
            table.add_row("Drop Columns", ", ".join(map(str, self.drop_columns)))

        table.add_row("Categorical Imputation", self.categorical_imputation_method)
        table.add_row("Numerical Imputation", self.numerical_imputation_method)
        if self.categorical_imputation_method == "constant":
            table.add_row("Categorical Imputation Const", str(self.categorical_imputation_constant))
        if self.numerical_imputation_method == "constant":
            table.add_row("Numerical Imputation Const", str(self.numerical_imputation_constant))
        table.add_row("Encoding Method", self.encoding_method)
        table.add_row("One-Hot Limit", str(self.onehot_limit))
        table.add_row("Normalize", str(self.normalize) if self.normalize else "None")
        table.add_row("Shuffle", str(self.shuffle))
        table.add_row("Logging to File", str(self.logging_to_file))

        # Conditionally add
        if self.column_imputation_map:
            table.add_row("Column Imputation Map", str(self.column_imputation_map))
        if self.encoding_method_map:
            table.add_row("Encoding Map", str(self.encoding_method_map))
        if self.ordinal_encode_map:
            table.add_row("Ordinal Encode Map", str(self.ordinal_encode_map))

        # Print the table to console (if using a console, such as in Jupyter)
        console = Console()
        console.print(table)

        return ""

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
    
    def __prepare_holdout_data(self, test_size: Optional[float] = None):
        """
        Prepares the holdout data for the model training process
        """
        holdout_cv_splits = get_cv_splits(
            df=self.data,
            cv_method="holdout",
            test_size=test_size,
            shuffle=self.shuffle,
            random_state=self._data_processing_random_state,
            ml_task_type=self.__ML_TASK_TYPE
        )[0]
        train_labels, test_labels = holdout_cv_splits[0], holdout_cv_splits[1]

        train_data = pd.concat([
            self.X.loc[train_labels], 
            self.y.loc[train_labels]
        ], axis=1)
        test_data = pd.concat([
            self.X.loc[test_labels],
            self.y.loc[test_labels]
        ], axis=1)

        self.feature_engineer.setup(data=train_data)

        self.X_train, self.y_train = self.feature_engineer.fit_transform()
        self.X_test, self.y_test = self.feature_engineer.transform(test_data=test_data, y_included=True)
        self.feature_names = list(self.X_train.columns)
        self.y_class_mapping = self.feature_engineer.y_class_mapping
        
    def __prepare_models(self, experiment_size: str, num_class: int, random_state: Optional[int] = None, n_jobs: Optional[int] = -1):
        """
        Prepares the models based on the selected experiment size ('quick' or 'wide')

        Parameters
        ----------
        experiment_size : str
            The size of the experiment to run. It can be 'quick' or 'wide'

        num_class : int
            The number of classes (If It's a Classification problem)

        random_state : int, optional (default=None)
            The random state value for the model training process

        n_jobs : int, optional (default=-1)
            The number of jobs to run in parallel. -1 means using all processors
        """
        if experiment_size not in ['quick', 'wide']:
            error_msg = f"experiment_size expected to be either 'quick' or 'wide', got {experiment_size}"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.__ML_MODELS = get_ml_models(
            ml_task_type=self.__ML_TASK_TYPE,
            num_class=num_class,
            random_state=random_state,
            n_jobs=n_jobs
        ).get(experiment_size.upper())
    
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
            
            # Calculate the average for all aggregated metrics, with special cases for "Time (sec)" and "Full Train"
            averaged_metrics = {
                key: (np.sum(value) if key == "Time (sec)" else 
                      value[0] if key == "Full Train" else 
                      np.mean(value) if isinstance(value[0], (int, float)) else value[0])
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
        groups_col: Optional[str] = None,
        n_jobs: Optional[int] = -1
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

        eval_metric : str, optional (default='R2' for Regression, 'Accuracy' for Classification)
            The evaluation metric to use for model evaluation
            
            - Avaiable evalulation metrics for Regression:    
                - R2, MAE, MSE, RMSE, MAPE

            - Avaiable evalulation metrics for Classification:    
                - Accuracy, Precision, Recall, F1 Score, ROC-AUC

        random_state : int, optional (default=None)
            The random state value for the model training process

        groups_col : str, optional
            Column name for group-based cross-validation methods

        n_jobs : int, optional (default=-1)
            The number of jobs to run in parallel. -1 means using all processors

        Notes for Cross-Validation Methods
        ----------------------------------
        - Group-based methods require `groups_col` to define group labels
        - If both `n_folds` and `test_size` are provided, shuffle-based methods are prioritized
        - Defaults to a standard 5-fold if neither `n_folds` nor `test_size` is provided
        """
        experiment_size = experiment_size.lower() # Convert to lowercase in case of any case mismatch
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
        if cv_method != "holdout" and n_folds is None:
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

            self.__prepare_holdout_data(test_size=test_size if cv_method == "holdout" else None)

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

        self.__prepare_models(experiment_size, self.num_class, random_state, n_jobs)
        cv_splits_copy = self.cv_splits.copy() # Will be used for trainings

        self.__logger.info(f"[PROCESS] Training the ML models with {cv_method} validation")

        # Train only the models that haven't been trained yet
        if not reset_the_experiment and quick_to_wide_flag: 
            self.__existing_model_names = list(self._model_stats_df['Model Name'].unique()) if self._model_stats_df is not None else []
        else:
            self.__model_training_info = []
            self._model_stats_df = None
            self.__existing_model_names = []
            self.__models_raised_error = []
            self._holdout_model_objects = {}
        
        all_model_stats = defaultdict(list)
        total_iterations = len(cv_splits_copy) * len(self.__ML_MODELS)

        with tqdm(
            total=total_iterations,
            desc="INFO | Training Progress",
            bar_format="{desc}: |{bar}| {percentage:.0f}%"
        ) as pbar:
            for train_idx, test_idx in cv_splits_copy:
                try:
                    # Attempt to use as positional indices (For Cross-Validation splits)
                    train_labels = self.X.index[train_idx]
                    test_labels = self.X.index[test_idx]
                except IndexError: # For holdout validation
                    # Fallback to using as label-based indices
                    train_labels = train_idx
                    test_labels = test_idx
                
                train_data = pd.concat([
                    self.X.loc[train_labels], 
                    self.y.loc[train_labels]
                ], axis=1)
                test_data = pd.concat([
                    self.X.loc[test_labels],
                    self.y.loc[test_labels]
                ], axis=1)
                
                self.feature_engineer.setup(data=train_data)
                
                X_train, y_train = self.feature_engineer.fit_transform()
                X_test, y_test = self.feature_engineer.transform(test_data=test_data, y_included=True)

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
                        if self.__ML_TASK_TYPE == "Classification" and hasattr(model, 'predict_proba'):
                            y_pred = model.predict_proba(X_test)
                        else:
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
                                "Model Name": model_name,
                                "Full Train": False,
                                **avg_metrics,
                                "Time (sec)": total_time_taken
                            }
                        })

                    except Exception as e:
                        # TODO: FlexML should add a Pipeline step for revising column names. Reference: https://stackoverflow.com/questions/60582050/lightgbmerror-do-not-support-special-json-characters-in-feature-name-the-same/62364946#62364946
                        if model_name == "LGBMClassifier" and str(e) == 'Do not support special JSON characters in feature name.':
                            self.__logger.error("LGBMClassifier does not support special characters in row/column names. Please make sure that your row/column names are consisted of *only* English characters")
                        else:
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
        if self.__model_training_info is None or len(self.__model_training_info) == 0:
            error_msg = "No models have been trained yet! Please start an experiment first via start_experiment()"
            self.__logger.error(error_msg)
            raise Exception(error_msg)

        for model_info in self.__model_training_info:
            if model_name in model_info.keys():
                return model_info[model_name]["model"]
        
        error_msg = f"{model_name} is not found in the trained models, expected one of the following:\n{[list(model_info.keys())[0] for model_info in self.__model_training_info]}"
        self.__logger.error(error_msg)
        raise ValueError(error_msg)


    def get_best_models(self, eval_metric: Optional[str] = None, top_n_models: int = 1) -> Union[object, list[object], None]:
        """
        Returns the top n models based on the evaluation metric.

        Parameters
        ----------
        top_n_models : int
            The number of top models to select based on the evaluation metric

        eval_metric : str, optional
            Default: eval_metric passed to the start_experiment(), If It was also None, 'R2' for Regression and 'Accuracy' for Classification will be used
        
            - Avaiable evalulation metrics for Regression:    
                - R2, MAE, MSE, RMSE, MAPE

            - Avaiable evalulation metrics for Classification:    
                - Accuracy, Precision, Recall, F1 Score, ROC-AUC
        
        Returns
        -------
        object or list[object] or None
            Single or a list of top n models based on the evaluation metric or None If no models have been trained yet.
        """
        if len(self.__model_training_info) == 0:
            return None
        
        top_n_models = self.__top_n_models_checker(top_n_models)

        if eval_metric is None and hasattr(self, 'eval_metric'):
            eval_metric = self.eval_metric
        eval_metric = eval_metric_checker(self.__ML_TASK_TYPE, eval_metric)
        
        model_stats = []
        best_models = []
        
        for model_pack in self.__model_training_info:
            for model_name, model_data in model_pack.items():
                model_stats.append(model_data["model_stats"])
    
        self._model_stats_df = pd.DataFrame(model_stats)
        self.__sorted_model_stats_df = self.__sort_models(eval_metric)

        for i in range(top_n_models):
            searched_model_name = self.__sorted_model_stats_df.iloc[i]["Model Name"]
            for model_info in self.__model_training_info:
                model_name = list(model_info.keys())[0]
                if model_name == searched_model_name:
                    best_models.append(model_info[model_name]["model"])
                    self.__last_searched_model_name = model_name # Save the model name so that If funcs like save_model() or predict() are called with model=None, FlexML will know the retrieved model object's "label"
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
        model : str or object, optional
            The name of the model to save or the model object itself
            If None, the best model will be fetched
        save_path : str, optional
            The path to save the pipeline
        include_feature_pipeline : bool, optional
            Whether to include the feature engineering pipeline in the saved pipeline
        full_train : bool, optional
            Whether to train the model using the whole data
            
        Returns
        -------
        Pipeline or object
        """
        model_taken_from_leaderboard = False # If the model object is from leaderboard, track this
        # Ensure save_path is defined
        if save_path is None:
            save_path = "pipeline.pkl" if not model_only else "model.pkl"
            self.__logger.info(f"No save path provided. Using default: {save_path}")

        if not save_path.endswith(".pkl"):
            self.__logger.warning(f"Only .pkl files are supported. Changing '{save_path}' to '{save_path.rsplit('.', 1)[0]}.pkl'.")
            save_path = save_path.rsplit('.', 1)[0] + ".pkl"

        # Fetch the best model if no specific model is provided
        if model is None:
            model = self.get_best_models()
            model_name = self.__last_searched_model_name
            if model is None:
                error_msg = "There is no model to save! Please start an experiment first via start_experiment()"
                self.__logger.error(error_msg)
                raise Exception(error_msg)
            model_taken_from_leaderboard = True

        elif isinstance(model, str):
            try:
                model_name = model
                model = self.get_model_by_name(model)
                model_taken_from_leaderboard = True
            except KeyError:
                error_msg = f"Model with name '{model}' not found."
                self.__logger.error(error_msg)
                raise ValueError(error_msg)
        else: # If model is an object, we can't know its name, so we use its class name
            model_name = model.__class__.__name__
            
        # Initialize pipeline steps
        pipeline_steps = []

        # Initialize and setup feature engineering if needed
        if not model_only:
            # Add the feature engineering pipeline directly
            pipeline_steps.extend(self.feature_engineer.pipeline.steps)

        # Handle full training scenario if required
        if full_train:
            already_trained = self._check_if_model_is_full_trained(model_name, model_taken_from_leaderboard)
            if not already_trained:
                self.__logger.info("Training the model using the whole data")
                self.feature_engineer.setup(data=self.data)
                X_train, y_train = self.feature_engineer.fit_transform()
                model.fit(X_train, y_train)

                # find the model in leaderboard and update the full_train to True, and update the model object in there
                for model_info in self.__model_training_info:
                    for name, info in model_info.items():
                        if name == model_name:
                            info["model_stats"]["Full Train"] = True
                            info["model"] = model
                            break
                # Update leaderboard
                self.get_best_models()

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
    
    def _check_if_model_is_full_trained(self, model_name: str, model_taken_from_leaderboard: bool) -> bool:
        """
        Checks if the model is full trained

        Parameters
        ----------
        model_name : str
            The name of the model

        model_taken_from_leaderboard : bool
            Whether the model is taken from leaderboard or user is passed a different model object

        Returns
        -------
        bool
            True if the model is full trained, False otherwise
        """
        if not model_taken_from_leaderboard:
            return False
        
        for model_info in self.__model_training_info:
            for name, info in model_info.items():
                if name == model_name and info["model_stats"].get("Full Train", False):
                    return True
        return False

    def _predict_helper(
        self,
        test_data: pd.DataFrame,
        model: Optional[Union[str, object]] = None,
        full_train: bool = True
    ) -> tuple:
        """Inner handler for prediction methods that returns prepared model and transformed data"""
        if test_data is None or not isinstance(test_data, pd.DataFrame) or test_data.empty:
            raise ValueError("test_data must be provided as a pandas DataFrame and non-empty")

        # Check column consistency
        drop_columns = set(self.feature_engineer.drop_columns)
        expected_columns = set(self.X.columns) - drop_columns
        test_columns = set(test_data.columns) - drop_columns

        if expected_columns != test_columns:
            missing = expected_columns - test_columns
            extra = test_columns - expected_columns
            error_msg = "Mismatch in test_data columns."
            if missing: error_msg += f" Missing: {missing}."
            if extra: error_msg += f" Extra: {extra}."
            raise ValueError(error_msg)
        
        model_taken_from_leaderboard = False # If the model object is from leaderboard, track this

        if model is None:
            model = self.get_best_models()
            model_name = self.__last_searched_model_name
            if model is None:
                error_msg = "There is no model to use for prediction! Please start an experiment first via start_experiment()"
                self.__logger.error(error_msg)
                raise Exception(error_msg)
            model_taken_from_leaderboard = True
        elif isinstance(model, str):
            model_name = model
            model = self.get_model_by_name(model)
            model_taken_from_leaderboard = True
        else: # If model is an object, we can't know its name, so we use its class name
            model_name = model.__class__.__name__
        
        # Prepare training data if needed
        if full_train:
            # Check If model_taken_from_leaderboard is True and Full Train in self.__model_training_info is True, then we don't need to train the model again
            already_trained = self._check_if_model_is_full_trained(model_name, model_taken_from_leaderboard)
            if not already_trained:
                self.__logger.info("Training the model using the whole data")
                self.feature_engineer.setup(data=self.data)
                X_train, y_train = self.feature_engineer.fit_transform()
                model.fit(X_train, y_train)

                # find the model in leaderboard and update the full_train to True, and update the model object in there
                for model_info in self.__model_training_info:
                    for name, info in model_info.items():
                        if name == model_name:
                            info["model_stats"]["Full Train"] = True
                            info["model"] = model
                            break
                # Update leaderboard
                self.get_best_models()
            X_test = self.feature_engineer.transform(test_data)

        else:
            X_test = self.feature_engineer.transform(test_data)

        return model, X_test

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
            The name of the model to predict or the model object itself
            If None, the best model will be fetched
        full_train : bool, optional
            Whether to train the model using the whole data before prediction

        Returns
        -------
        np.ndarray
            The predicted target column
        """
        model, X_test = self._predict_helper(test_data, model, full_train)
        try:
            predictions = self.feature_engineer.target_encoder.inverse_transform(model.predict(X_test))
        except AttributeError:
            predictions = model.predict(X_test)

        return predictions
    
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
            Whether to train the model using the whole data before prediction

        Returns
        -------
        np.ndarray
            The predicted probabilities for each class
        """
        if self.__ML_TASK_TYPE == "Regression":
            error_msg = "predict_proba is not available for regression tasks"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        model, X_test = self._predict_helper(test_data, model, full_train)
        return model.predict_proba(X_test)

    def __get_holdout_model_from_stats(self, model_name: str) -> object:
        if self._holdout_model_objects is None or self._holdout_model_objects == {}:
            return None
        return self._holdout_model_objects.get(model_name)
    
    def __add_holdout_model_to_stats(self, model: object, model_name: Optional[str] = None):
        if model_name is None:
            model_name = model.__class__.__name__

        model_copy = deepcopy(model)
        model_copy.fit(self.X_train, self.y_train)
        self._holdout_model_objects[model_name] = model_copy
        return model_copy
    
    def plot(self, model: Optional[Union[str, object]] = None, kind: str = "feature_importance", **kwargs):
        """
        Plots the model performance graphs

        - For Regression:
            - "feature_importance"
            - "residuals"
            - "prediction_error"
            - "calibration_curve"
            - "shap_summary"
            - "shap_violin"
            
        - For Classification:
            - "feature_importance"
            - "confusion_matrix"
            - "roc_curve"
            - "calibration_curve"
            - "shap_summary"
            - "shap_violin"

        **Warning:**
        
        The outputs of the plots may not be equal with the results in the leaderboard If you used `cv_method` other than "holdout".

        The reason is FlexML can't hold all X, y pairs across all folds and predictions of each model for each fold since It would cause high memory usage and time cost for you,
        In order to generalize the results, FlexML uses 25% of the data for holdout validation and train the model on the rest of the data and show the results of the holdout validation. 
    
        If you have used "holdout" as `cv_method`, generated holdout will be based on the `test_size` param you have passed to the start_experiment() method. So, the results will be the same as the leaderboard

        Parameters
        ----------
        model : str or object, optional
            The name of the model to plot or the model object itself
            If None, the best model will be fetched

        kind : str, optional
            The type of the plot to plot
        
        **kwargs : dict or param=value pair, optional
            Additional keyword arguments to pass to the plot function

            `width` and `height` in pixels (default = 800 and 600) (Supported for "feature_importance", "confusion_matrix", "roc_curve", "calibration_curve")

            - "feature_importance"
                - `top_x_features` : int (default = 10)
                    The number of top features to display in the plot

            - "calibration_curve"
                - `n_bins` : int (default = 10)
                    The number of bins to discretize the [0, 1] interval
                - `strategy` : str (default = 'uniform')
                    The strategy used to define the widths of the bins
                    - "uniform" : The bins have equal width
                    - "quantile" : The bins have equal number of points
        """
        available_plot_types = PLOT_TYPES.get(self.__ML_TASK_TYPE, [])

        if kind not in available_plot_types:
            error_msg = f"Invalid plot type: {kind}. Available plot kinds: {available_plot_types}"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        if model is None:
            model = self.get_best_models()
            model_name = self.__last_searched_model_name
        elif isinstance(model, str):
            model_name = model
            model = self.get_model_by_name(model)
        else: # If model is an object, we can't know its name, so we use its class name
            model_name = model.__class__.__name__
        
        if self.__get_holdout_model_from_stats(model_name) is not None:
            model = self.__get_holdout_model_from_stats(model_name)
        else:
            model = self.__add_holdout_model_to_stats(model, model_name)

        # If kind expects predictions
        if kind in ["confusion_matrix"]:
            preds = model.predict(self.X_test)
        elif kind in ["roc_curve", "calibration_curve"]:
            preds = model.predict_proba(self.X_test)

        graph = None

        if kind == "feature_importance":
            if not hasattr(self, 'feature_names'):
                self.feature_names = list(self.X_train.columns)
            graph = plot_feature_importance(model, self.feature_names, **kwargs)
        elif kind == "confusion_matrix":
            graph = plot_confusion_matrix(self.y_test, preds, self.y_class_mapping, **kwargs)
        elif kind == "roc_curve":
            graph = plot_roc_curve(self.y_test, preds, self.y_class_mapping, **kwargs)
        elif kind == "residuals":
            graph = plot_residuals(model, self.X_train, self.y_train, self.X_test, self.y_test, **kwargs)
        elif kind == "prediction_error":
            graph = plot_prediction_error(model, self.X_train, self.y_train, self.X_test, self.y_test, **kwargs)
        elif kind == "calibration_curve":
            graph = plot_calibration_curve(self.y_test, preds, self.y_class_mapping, **kwargs)
        elif 'shap' in kind:
            graph = plot_shap(model, self.X_test, kind, **kwargs)
        else:
            error_msg = f"Invalid plot type: {kind}. Available plot types: {available_plot_types}"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        if isinstance(graph, str):
            self.__logger.error(graph)
            raise ValueError(graph)
        
        if graph is not None and not isinstance(graph, bool):
            graph.show()
        elif graph is not None and graph != True:
            error_msg = f"Failed to plot {kind} for the model {model} due to an unknown error"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
    def __sort_models(self, eval_metric: Optional[str] = None):
        """
        Sorts the models based on the evaluation metric.

        Parameters
        ----------
        eval_metric : str, optional
            Default: eval_metric passed to the start_experiment(), If It was also None, 'R2' for Regression and 'Accuracy' for Classification will be used
        
            - Avaiable evalulation metrics for Regression:    
                - R2, MAE, MSE, RMSE, MAPE

            - Avaiable evalulation metrics for Classification:    
                - Accuracy, Precision, Recall, F1 Score, ROC-AUC

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the sorted model statistics according to the desired eval_metric
        """
        if self._model_stats_df is None or len(self._model_stats_df) == 0:
            error_msg = "There is no model performance data to sort!"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Since lower is better for mae, mse and rmse in Regression tasks, they should be sorted in ascending order
        if self.__ML_TASK_TYPE == "Regression" and eval_metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
            return self._model_stats_df.sort_values(by=eval_metric, ascending=True).reset_index(drop = True)
        else:
            return self._model_stats_df.sort_values(by=eval_metric, ascending=False).reset_index(drop = True)

    def show_model_stats(self, eval_metric: Optional[str] = None):
        """
        Sorts and shows the model statistics table based on the evaluation metric.

        Parameters
        ----------
        eval_metric : str, optional
            Default: eval_metric passed to the start_experiment(), If It was also None, 'R2' for Regression and 'Accuracy' for Classification will be used
        
            - Avaiable evalulation metrics for Regression:    
                - R2, MAE, MSE, RMSE, MAPE

            - Avaiable evalulation metrics for Classification:    
                - Accuracy, Precision, Recall, F1 Score, ROC-AUC
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
                if best_val == float('inf'):
                    is_best = pd.Series([False] * len(s))
                else:
                    is_best = (s == best_val) & (s != float('inf')) & (s != -1)
            else:
                is_best = (s == s.max()) & (s != float('inf')) & (s != -1)
            return ['background-color: green' if v else '' for v in is_best]
        
        
        if eval_metric is None and hasattr(self, 'eval_metric'):
            eval_metric = self.eval_metric
        eval_metric = eval_metric_checker(self.__ML_TASK_TYPE, eval_metric)

        sorted_model_stats_df = self.__sort_models(eval_metric)
        sorted_model_stats_df['Time (sec)'] = sorted_model_stats_df['Time (sec)'].apply(lambda x: f"{x:.2f}")
        sorted_model_stats_df.index += 1
        sorted_model_stats_df = sorted_model_stats_df.drop('Full Train', axis=1)
        
                
        # Check if we're in an interactive notebook environment (Jupyter or Colab)
        if not is_interactive_notebook():
            print(100*'-')
            print(sorted_model_stats_df.head(len(self.__ML_MODELS)))
            print(100*'-')
        else:
            # Apply the highlighting to all metric columns and display the dataframe If It has more than 1 row so that we can compare the models
            if len(sorted_model_stats_df) < 2:
                display(sorted_model_stats_df)
            else:
                styler = sorted_model_stats_df.style.apply(highlight_best, subset=self.__ALL_EVALUATION_METRICS)
                display(styler) # display is only supported in interactive kernels such as Jupyter Notebook/Google Colab

    def tune_model(
        self, 
        model: Optional[Union[object, str]] = None,
        tuning_method: str = 'randomized_search',
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
        model : object or str (default = None)
            The machine learning model to tune. If It's none, flexml retrieves the best model found in the experiment
            If It's a string, flexml will try to get the model from the model list
            
        tuning_method: str (default = 'randomized_search')
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

        eval_metric : str, optional
            Default: eval_metric passed to the start_experiment(), If It was also None, 'R2' for Regression and 'Accuracy' for Classification will be used
        
            - Avaiable evalulation metrics for Regression:    
                - R2, MAE, MSE, RMSE, MAPE

            - Avaiable evalulation metrics for Classification:    
                - Accuracy, Precision, Recall, F1 Score, ROC-AUC

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
            if tuning_report is None or tuning_report.get('model_perf') is None:
                return False
            
            self.tuned_model = tuning_report['tuned_model']
            self.tuned_model_score = tuning_report['tuned_model_score']
            tuned_time_taken = tuning_report['time_taken_sec']
            base_model_name = f"{self.tuned_model.__class__.__name__}_({tuning_report['tuning_method']})_(n_iter={tuning_report['n_iter']})"
            
            # Find all existing models with same base name (including those with suffixes)
            max_suffix = 0
            for key in self.__existing_model_names:
                # Check if it's the exact base name or a suffixed version
                if key == base_model_name:
                    max_suffix = max(max_suffix, 1)
                elif key.startswith(f"{base_model_name}_"):
                    try:
                        suffix_num = int(key.split("_")[-1])
                        max_suffix = max(max_suffix, suffix_num + 1)
                    except (ValueError, IndexError):
                        pass
            
            # Determine the model name
            if max_suffix > 0:
                tuned_model_name = f"{base_model_name}_{max_suffix}"
            else:
                tuned_model_name = base_model_name
            
            # Add the model name to existing_model_name dict
            self.__existing_model_names.append(tuned_model_name)

            # Add the tuned model and it's score to the model_training_info list
            model_perf = tuning_report['model_perf']
            self.__model_training_info.append({
                tuned_model_name:{
                    "model": self.tuned_model,
                    "model_stats": {
                        "Model Name": tuned_model_name,
                        "Full Train": True if tuning_method != "optuna" else False, # refit is done in grid_search and randomized_search, but not in optuna
                        **model_perf,
                        "Time (sec)": tuned_time_taken
                    }
                }
            })
            self.get_best_models() # Update the self._model_stats_df
            self.show_model_stats()
            return True
        
        if self._model_stats_df is None or len(self._model_stats_df) == 0:
            error_msg = "Model leaderboard is empty! Please start an experiment first via start_experiment()"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not isinstance(model, object) and not isinstance(model, str):
            error_msg = f"model parameter should be an object or a string, got {type(model)}"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        if isinstance(model, str):
            model = self.get_model_by_name(model)
        
        if tuning_method not in ['grid_search', 'randomized_search', 'optuna']:
            error_msg = f"tuning_method parameter should be one of the following: 'grid_search', 'randomized_search', 'optuna', got {tuning_method}"
            self.__logger.error(error_msg)
            raise ValueError(error_msg)
        
        if eval_metric is None and hasattr(self, 'eval_metric'):
            eval_metric = self.eval_metric
        eval_metric = eval_metric_checker(self.__ML_TASK_TYPE, eval_metric)
        
        # If the user doesn't pass any cross-validation method params, use the last used ones
        if (
            cv_method is None and hasattr(self, '_last_cv_method') and \
            n_folds is None and hasattr(self, '_last_n_folds') and \
            test_size is None and hasattr(self, '_last_test_size') and \
            groups_col is None and hasattr(self, '_last_groups_col')
        ):
            cv_method = self._last_cv_method
            n_folds = self._last_n_folds
            test_size = self._last_test_size
            groups_col = self._last_groups_col

        else:
            cv_method = cross_validation_checker(
                df=self.data,
                cv_method=cv_method,
                n_folds=n_folds,
                test_size=test_size,
                groups_col=groups_col,
                available_cv_methods=self.__AVAILABLE_CV_METHODS,
                ml_task_type=self.__ML_TASK_TYPE
            )
            if cv_method != "holdout" and n_folds is None:
                n_folds = 5

        # Get the best model If the user doesn't pass any model object
        if model is None:
            model = self.get_best_models(eval_metric)
            if model is None:
                error_msg = "There is no model to tune! Please start an experiment first via start_experiment()"
                self.__logger.error(error_msg)
                raise Exception(error_msg)

        # Get the model's param_grid from the config file If It's not passed from the user
        if param_grid is None:
            try:
                param_grid = [ml_model for ml_model in self.__ML_MODELS if ml_model['name'] == model.__class__.__name__][0]['tuning_param_grid']

            except IndexError:
                error_msg = f"{model}'s tuning param_grid is not found (Is it a model not located in the model leaderboard?), please then pass it manually via 'param_grid' parameter"
                self.__logger.error(error_msg)
                raise ValueError(error_msg)
                
            except AttributeError:
                error_msg = f"{model}'s tuning param_grid is not found in the config, please pass it manually via 'param_grid' parameter"
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
            if self._model_stats_df is not None:
                self.__logger.warning("Validation params (e.g. cv_method, n_folds, test_size, groups_col) you've provided are different than the last run. Model performance table will be erased")
                self._model_stats_df = None
                self.__model_training_info = []
                self.__existing_model_names = []
                self._holdout_model_objects = {}

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
            if self.__ML_TASK_TYPE == 'Classification' and self.y.dtype in ['object', 'category']:
                y_encoded = pd.Series(self.feature_engineer.target_encoder.fit_transform(self.y), name=self.target_col)
                y_encoded.index = self.y.index
            else:
                y_encoded = self.y # No need to encode the target for regression or if the target is already encoded
            self.model_tuner = ModelTuner(self.__ML_TASK_TYPE, self.X, y_encoded, self.logging_to_file)

        pipeline = self.feature_engineer.pipeline
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
            tuning_result = self.model_tuner.randomized_search(
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
        
        if _show_tuning_report(tuning_result):
            self.__logger.info("[PROCESS] Model Tuning process is finished successfully")
        else:
            self.__logger.warning("Model Tuning process is failed, Please check the error messages appeared")