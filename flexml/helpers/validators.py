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
        All possible evaluation metrics for the current task (Regression or Classification), e.g. ['R2', 'MAE', 'MSE', 'RMSE'] for Regression

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

    if not isinstance(random_state, int) or random_state < 0:
        error_msg = f"random_state should be a positive integer, got {random_state}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return random_state