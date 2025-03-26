import pandas as pd
from typing import Optional, List
from flexml.config import EVALUATION_METRICS, FEATURE_ENGINEERING_METHODS, CROSS_VALIDATION_METHODS
from flexml.logger import get_logger
import re

def eval_metric_checker(
    ml_task_type: str,
    eval_metric: Optional[str] = None,
    all_evaluation_metrics: Optional[List[str]] = None,
    default_evaluation_metric: Optional[str] = None
) -> str:
    """
    Since eval_metric setting and validation is a common process for both Regression and Classification tasks...
    this method is used to set and validate the evaluation metric.

    Parameters
    ----------
    ml_task_type : str
        The type of ML task ('Regression' or 'Classification')

    eval_metric : str, optional (default='R2' for Regression, 'Accuracy' for Classification)
        The evaluation metric to use for model evaluation

        - Avaiable evalulation metrics for Regression:    
            - R2, MAE, MSE, RMSE, MAPE

        - Avaiable evalulation metrics for Classification:    
            - Accuracy, Precision, Recall, F1 Score, ROC-AUC
    
    all_evaluation_metrics : List[str], (default=None)
        All possible evaluation metrics for the current task (Regression or Classification), e.g. ['R2', 'MAE', 'MSE', 'RMSE', 'MAPE'] for Regression

        If passed as None, they will be fetched from the config file

    default_evaluation_metric : str, (default=None)
        The default evaluation metric to use for the current task (Regression or Classification) e.g. 'R2' for Regression, 'Accuracy' for Classification

        If passed as None, it will be fetched from the config file

    Returns
    -------
    str
        The evaluation metric to use for model evaluation for the current task (Regression or Classification)
    """
    logger = get_logger(__name__, "PROD", False)
    
    if default_evaluation_metric is None or all_evaluation_metrics is None:
        default_evaluation_metric = EVALUATION_METRICS[ml_task_type]["DEFAULT"]
        all_evaluation_metrics = EVALUATION_METRICS[ml_task_type]["ALL"]

    if eval_metric is None:
        return default_evaluation_metric
    
    if ml_task_type == "Regression":
        eval_metric = eval_metric.upper()
    else:
        # Normalize input for flexible matching
        original_metric = eval_metric
        normalized_input = re.sub(r'[^a-zA-Z0-9]', '', eval_metric).lower()
        normalized_config = {re.sub(r'[^a-zA-Z0-9]', '', m).lower(): m 
                            for m in all_evaluation_metrics}
        
        if normalized_input in normalized_config:
            eval_metric = normalized_config[normalized_input]
        else:
            error_msg = (f"'{original_metric}' is not a valid evaluation metric for {ml_task_type}, "
                        f"expected one of: {all_evaluation_metrics}")
            logger.error(error_msg)
            raise ValueError(error_msg)

    if eval_metric not in all_evaluation_metrics:
        error_msg = f"Validation failed for {eval_metric} - not in configured metrics"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return eval_metric

def random_state_checker(random_state: Optional[int] = None) -> int:
    """
    Validates the random_state parameter

    Parameters
    ----------
    random_state : int, optional (default=None)
        Random state value

    Returns
    -------
    int
        Validated random state
    """
    logger = get_logger(__name__, "PROD", False)

    if random_state is not None and (not isinstance(random_state, int) or random_state < 0):
        error_msg = f"random_state should be either None or a positive integer, got {random_state}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return random_state

def cross_validation_checker(
    df: pd.DataFrame,
    cv_method: Optional[str] = None,
    n_folds: Optional[int] = None,
    test_size: Optional[float] = None,
    groups_col: Optional[str] = None,
    available_cv_methods: Optional[dict] = None,
    ml_task_type: Optional[str] = None
) -> str:
    
    """
    df : pd.DataFrame
        The DataFrame that will be performed cross-validation to

    cv_method : str, (default='kfold' for Regression, 'stratified_kfold' for Classification if ml_task_type is not None)
        The cross-validation method to use

        If passed as None, the default cross-validation method for the corresponding ml_task_type will be used If ml_task_type is not None

    n_folds : int, optional (default=None)
        Number of folds to use for cross-validation

    test_size : float, optional (default=None)
        The proportion of the dataset to include in the test split

    groups_col : str, optional (default=None)
        The column in the DataFrame that contains the groups for group-based cross-validation methods
    
    available_cv_methods : dict, optional (default=None)
        A dictionary containing the available cross-validation methods

    ml_task_type : str, optional (default=None)
        The type of ML task ('Regression' or 'Classification')

    Returns
    -------
    str
        The cross-validation method to use for the current task (Regression or Classification)
    """
    logger = get_logger(__name__, "PROD", False)

    if ml_task_type is not None and ml_task_type not in ['Regression', 'Classification']:
        error_msg = f"ml_task_type should be 'Regression' or 'Classification', got {ml_task_type}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if available_cv_methods is None:
        if ml_task_type is not None:
            available_cv_methods = CROSS_VALIDATION_METHODS[ml_task_type]
        else:
            available_cv_methods = CROSS_VALIDATION_METHODS['all']

    if cv_method is None:
        if ml_task_type is not None:
            if ml_task_type == 'Regression':
                cv_method = 'kfold'
            elif ml_task_type == "Classification":
                cv_method = 'stratified_kfold'

    else:
        cv_method = cv_method.lower()
        if available_cv_methods.get(cv_method) is None:
            # If cv_method is not found in the available cv methods, check the without '_' version -->
            # e.g. 'stratified_kfold' and 'stratifiedkfold'
            if cv_method in available_cv_methods.values():
                cv_method = list(available_cv_methods.keys())[list(available_cv_methods.values()).index(cv_method)]
    
    # Check if cv_method is still None
    if cv_method is None or cv_method not in list(available_cv_methods.keys()):
        error_msg = f"cv_method is not found in the available cross-validation methods, expected one of {list(available_cv_methods.keys())}, got {cv_method}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if n_folds is not None and (not isinstance(n_folds, int) or n_folds < 2):
        error_msg = "`n_folds` must be an integer >= 2 if provided"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if test_size is not None and (not isinstance(test_size, float) or not 0 < test_size < 1):
        error_msg = f"test_size parameter expected to be a float between 0 and 1, got {test_size}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if groups_col is not None and groups_col not in df.columns:
        error_msg = f"groups_col should be a column in the DataFrame, got {groups_col}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if cv_method in ["group_kfold", "group_shuffle_split"] and groups_col is None:
        error_msg = "`groups_col` must be provided for group-based methods"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return cv_method

def validate_inputs(
    data: pd.DataFrame,
    target_col: str, 
    drop_columns=None,
    categorical_imputation_method="mode",
    numerical_imputation_method="mean",
    column_imputation_map=None,
    numerical_imputation_constant=0.0,
    categorical_imputation_constant="Unknown",
    encoding_method="label_encoder",
    onehot_limit=25,
    encoding_method_map=None,
    ordinal_encode_map=None,
    normalize=None
):
    """
    Validates the input parameters for the feature engineering process

    Parameters
    ----------
    data : pd.DataFrame
        The input data for the model training process
    
    target_col : str
        The target column name in the data

    drop_columns : list, default=None
        Columns that will be dropped from the data.
    
    categorical_imputation_method : str, default='mode'
        Imputation method for categorical columns. Options:
        * 'mode': Replace missing values with the most frequent value.
        * 'constant': Replace missing values with a constant value.
        * 'drop': Drop rows with missing values.

    numerical_imputation_method : str, default='mean'
        Imputation method for numerical columns. Options:
        * 'mean': Replace missing values with the column mean.
        * 'median': Replace missing values with the column median.
        * 'mode': Replace missing values with the column mode.
        * 'constant': Replace missing values with a constant value.
        * 'drop': Drop rows with missing values.

    column_imputation_map : dict, default=None
        Custom mapping of columns to specific imputation methods.
        Example usage: {'column_name': 'mean', 'column_name2': 'mode'}

    numerical_imputation_constant : float, default=0.0
        The constant value for imputing numerical columns when 'constant' is selected.

    categorical_imputation_constant : str, default='Unknown'
        The constant value for imputing categorical columns when 'constant' is selected.

    encoding_method : str, default='label_encoder'
        Encoding method for categorical columns. Options:
        * 'label_encoder': Use label encoding
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
        * 'onehot_encoder': Use one-hot encoding
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        * 'ordinal_encoder': Use ordinal encoding
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
        
    onehot_limit : int, default=25
        Maximum number of categories to use for one-hot encoding.

    encoding_method_map : dict, default=None
        Custom mapping of columns to encoding methods.
        Example usage: {'column_name': 'onehot_encoder', 'column_name2': 'label_encoder'}
    
    ordinal_encode_map : dict, default=None
        Custom mapping of columns to category order for ordinal encoding.
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
    """
    # Check if any of the columns in drop_columns match the target_col
    if drop_columns is not None and target_col in drop_columns:
        error_msg = f"The target column '{target_col}' cannot be in the drop_columns list"
        raise ValueError(error_msg)
    
    if drop_columns is None:
        drop_columns = []
    remaining_columns = set(data.columns) - set(drop_columns)

    # Ensure the target column is in the remaining columns and there's at least one feature column
    if target_col not in remaining_columns or len(remaining_columns) < 2:
        error_msg = (
            f"After dropping columns, only {remaining_columns} remain. "
            f"There should be at least one feature column and the target column '{target_col}' remaining."
        )
        raise ValueError(error_msg)
    
    # Check if categorical_imputation_method is valid
    if categorical_imputation_method not in FEATURE_ENGINEERING_METHODS["accepted_categorical_imputations_methods"]:
        error_msg = f"The categorical_imputation_method '{categorical_imputation_method}' is not valid. Expected one of the following: {FEATURE_ENGINEERING_METHODS['accepted_categorical_imputations_methods']}"
        raise ValueError(error_msg)
    
    # Check if numerical_imputation_method is valid
    if numerical_imputation_method not in FEATURE_ENGINEERING_METHODS["accepted_numeric_imputations_methods"]:
        error_msg = f"The numerical_imputation_method '{numerical_imputation_method}' is not valid. Expected one of the following: {FEATURE_ENGINEERING_METHODS['accepted_numeric_imputations_methods']}"
        raise ValueError(error_msg)
    
    # Check if encoding_method is valid
    if encoding_method not in FEATURE_ENGINEERING_METHODS["accepted_encoding_methods"]:
        error_msg = f"The encoding_method '{encoding_method}' is not valid. Expected one of the following: {FEATURE_ENGINEERING_METHODS['accepted_encoding_methods']}"
        raise ValueError(error_msg)
    
    # Check if onehot_limit is a positive integer
    if not isinstance(onehot_limit, int) or onehot_limit < 0:
        error_msg = f"onehot_limit should be a positive integer, got {onehot_limit}"
        raise ValueError(error_msg)
    
    # Check if drop_columns columns are in data
    if drop_columns is not None:
        for col in drop_columns:
            if col not in data.columns:
                error_msg = f"The column '{col}' in drop_columns is not in the data"
                raise ValueError(error_msg)
        
    # Check if columns in column_imputation_map are in data and methods are valid
    if column_imputation_map is not None:
        for col, method in column_imputation_map.items():
            if col not in data.columns:
                error_msg = f"The column '{col}' in column_imputation_map is not in the data"
                raise ValueError(error_msg)
            
            if col in data.select_dtypes(include=['number']).columns:
                if method not in FEATURE_ENGINEERING_METHODS["accepted_numeric_imputations_methods"]:
                    error_msg = f"The numeric imputation method '{method}' for column '{col}' is not valid. Expected one of the following: {FEATURE_ENGINEERING_METHODS['accepted_numeric_imputations_methods']}"
                    raise ValueError(error_msg)
            else:
                if method not in FEATURE_ENGINEERING_METHODS["accepted_categorical_imputations_methods"]:
                    error_msg = f"The categorical imputation method '{method}' for column '{col}' is not valid. Expected one of the following: {FEATURE_ENGINEERING_METHODS['accepted_categorical_imputations_methods']}"
                    raise ValueError(error_msg)

    # Check if numerical_imputation_constant is a number
    if not isinstance(numerical_imputation_constant, (int, float)):
        error_msg = f"numerical_imputation_constant should be a number, got {type(numerical_imputation_constant)}"
        raise ValueError(error_msg)

    # Check if categorical_imputation_constant is a string
    if not isinstance(categorical_imputation_constant, str):
        error_msg = f"categorical_imputation_constant should be a string, got {type(categorical_imputation_constant)}"
        raise ValueError(error_msg)

    # Check if encoding_method is ordinal_encoder and ordinal_encoder_map is provided for every categorical column
    if encoding_method == "ordinal_encoder":
        if ordinal_encode_map is None:
            error_msg = "Ordinal encoding is selected but no ordinal_encode_map is provided"
            raise ValueError(error_msg)
        # Check if ordinal_encode_map is provided for every categorical column
        for col in data.select_dtypes(include=['object', 'category']).columns:
            if col not in ordinal_encode_map:
                error_msg = f"Ordinal encoding is selected for column '{col}' but no ordinal_encode_map is provided"
                raise ValueError(error_msg)

    # Check if methods inside encoding_method_map are valid and columns are in data
    if encoding_method_map is not None:
        for col, method in encoding_method_map.items():
            if col not in data.columns:
                error_msg = f"The column '{col}' in encoding_method_map is not in the data"
                raise ValueError(error_msg)
            
            if col in drop_columns:
                error_msg = f"The column '{col}' in encoding_method_map is in drop_columns"
                raise ValueError(error_msg)

            if method not in FEATURE_ENGINEERING_METHODS["accepted_encoding_methods"]:
                error_msg = f"The encoding method '{method}' for column '{col}' is not valid. Expected one of the following: {FEATURE_ENGINEERING_METHODS['accepted_encoding_methods']}"
                raise ValueError(error_msg)

            # Check if there is a ordinal_encoder between methods and ordinal_encode_map is provided
            if method == "ordinal_encoder":
                if ordinal_encode_map is None:
                    error_msg = f"Ordinal encoding is selected for column '{col}' but no ordinal_encode_map is provided"
                    raise ValueError(error_msg)
                # Check if map for col is provided within ordinal_encode_map
                if col not in ordinal_encode_map:
                    error_msg = f"Ordinal encoding is selected for column '{col}' but no ordinal_encode_map is provided"
                    raise ValueError(error_msg)

    # Check if normalize is valid
    if normalize is not None and normalize not in FEATURE_ENGINEERING_METHODS["accepted_standardization_methods"]:
        error_msg = f"The normalize method '{normalize}' is not valid. Expected one of the following: {FEATURE_ENGINEERING_METHODS['accepted_standardization_methods']}"
        raise ValueError(error_msg)

    # Check if encoding_method is ordinal_encoder
    if encoding_method == "ordinal_encoder":
        if ordinal_encode_map is None:
            error_msg = "Ordinal encoding is selected, but no ordinal_encode_map is provided."
            raise ValueError(error_msg)
        
        # Get all categorical columns
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Check that all categorical columns are in ordinal_encode_map
        for col in categorical_columns:
            if col not in ordinal_encode_map:
                error_msg = f"Ordinal encoding is selected, but column '{col}' is missing in ordinal_encode_map."
                raise ValueError(error_msg)
            
            # Get distinct values in the column
            distinct_values = set(data[col].dropna().unique())
            map_values = set(ordinal_encode_map[col])
            
            # Check if the values in ordinal_encode_map match exactly with the distinct values
            if distinct_values != map_values:
                error_msg = (
                    f"Distinct values in column '{col}' do not match "
                    f"Ensure they match exactly."
                )
                raise ValueError(error_msg)
        
        # Check that ordinal_encode_map does not include extra columns
        extra_columns = set(ordinal_encode_map.keys()) - set(categorical_columns)
        if extra_columns:
            error_msg = (
                f"Ordinal_encode_map includes extra columns not in the categorical columns: {extra_columns}. "
                f"Remove these columns from the mapping."
            )
            raise ValueError(error_msg)

    # Check if encoding_method_map is provided and has ordinal_encoder
    if encoding_method_map:
        ordinal_columns = [
            col for col, method in encoding_method_map.items() if method == "ordinal_encoder"
        ]
    else:
        ordinal_columns = []

    if ordinal_columns:
        if not ordinal_encode_map:
            raise ValueError(
                "Ordinal encoding is specified in encoding_method_map, but no ordinal_encode_map is provided."
            )
        
        # Validate only the columns specified for ordinal encoding
        for col in ordinal_columns:
            if col not in ordinal_encode_map:
                raise ValueError(
                    f"Column '{col}' is marked for ordinal encoding but is missing in ordinal_encode_map."
                )
            
            # Get distinct values in the column
            distinct_values = set(data[col].dropna().unique())
            map_values = set(ordinal_encode_map[col])
            
            # Check if the values in ordinal_encode_map match exactly with the distinct values
            if distinct_values != map_values:
                raise ValueError(
                    f"Unique values in '{col}' do not match with the ones given in ordinal_encode_map. "
                    f"Ensure they match exactly."
                )
        
        # Ensure ordinal_encode_map does not include extra columns
        extra_columns = set(ordinal_encode_map.keys()) - set(ordinal_columns)
        if extra_columns:
            raise ValueError(
                f"Ordinal_encode_map includes extra columns not specified for ordinal encoding."
                f"Remove these columns from the mapping."
            )

    return True