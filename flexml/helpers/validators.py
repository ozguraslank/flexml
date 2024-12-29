import pandas as pd
from typing import Optional, List, Any
from flexml.config.supervised_config import EVALUATION_METRICS
from flexml.logger.logger import get_logger

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

    eval_metric : str, (default=None)
        The evaluation metric to use for model evaluation

        If passed as None, the default evaluation metric of the corresponding ml_task_type will be used
    
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
    
    if not isinstance(eval_metric, str):
        error_msg = f"eval_metric expected to be a string, got {type(eval_metric)}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    
    if ml_task_type == "Regression":
        eval_metric = eval_metric.upper()
    else:
        eval_metric = eval_metric.lower().capitalize()

    if eval_metric not in all_evaluation_metrics:
        error_msg = f"{eval_metric} is not a valid evaluation metric for {ml_task_type}, expected one of the following: {all_evaluation_metrics}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return eval_metric

def random_state_checker(random_state: Any) -> int:
    """
    Validates the random_state parameter

    Parameters
    ----------
    random_state : Any
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

    if cv_method is None:
        if ml_task_type is not None:
            if ml_task_type == 'Regression':
                cv_method = 'kfold'
            elif ml_task_type == "Classification":
                cv_method = 'stratified_kfold'
            else:
                error_msg = f"ml_task_type should be 'Regression' or 'Classification', got {ml_task_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)
    else:
        cv_method = cv_method.lower()
        if available_cv_methods is not None and isinstance(available_cv_methods, dict):
            if available_cv_methods.get(cv_method) is None:
                # If cv_method is not found in the available cv methods, check the without '_' version -->
                # e.g. 'stratified_kfold' and 'stratifiedkfold'
                if cv_method in available_cv_methods.values():
                    cv_method = list(available_cv_methods.keys())[list(available_cv_methods.values()).index(cv_method)]
        
    # Check if cv_method is still None
    if cv_method is None:
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
    
    if groups_col is not None and groups_col not in df.columns:
        error_msg = f"`groups_col` must be a column in `df`, got '{groups_col}'"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if cv_method in ["group_kfold", "group_shuffle_split"] and groups_col is None:
        error_msg = "`groups_col` must be provided for group-based methods"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return cv_method