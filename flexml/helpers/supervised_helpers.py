import numpy as np
import pandas as pd
from typing import Union

from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score)

def _evaluate_preds(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    eval_metric: str
) -> float:
    """
    Evaluates the model with the given evaluation metric by using the test set

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        The actual values of the target column

    y_pred : pd.Series or np.ndarray
        The predicted values of the target column

    eval_metric : str
        The evaluation metric that will be used to evaluate the model. It can be one of the following:
        
        * 'R2' for R^2 score
        
        * 'MAE' for Mean Absolute Error
        
        * 'MSE' for Mean Squared Error
        
        * 'RMSE' for Root Mean Squared Error
        
        * 'Accuracy' for Accuracy
        
        * 'Precision' for Precision
        
        * 'Recall' for Recall
        
        * 'F1 Score' for F1 score
        
        * 'MAPE' for Mean Absolute Percentage Error
        
        * 'ROC-AUC' for Receiver Operating Characteristic Area Under Curve

    Returns
    -------
    float
        The evaluation metric score for the desired eval metric
    """
    if eval_metric == 'R2':
        return round(r2_score(y_true, y_pred), 6)
    elif eval_metric == 'MAE':
        return round(mean_absolute_error(y_true, y_pred), 6)
    elif eval_metric == 'MSE':
        return round(mean_squared_error(y_true, y_pred), 6)
    elif eval_metric == 'RMSE':
        return round(np.sqrt(mean_squared_error(y_true, y_pred)), 6)
    elif eval_metric == 'Accuracy':
        return round(accuracy_score(y_true, y_pred), 6)
    elif eval_metric == 'Precision':
        return round(precision_score(y_true, y_pred), 6)
    elif eval_metric == 'Recall':
        return round(recall_score(y_true, y_pred), 6)
    elif eval_metric == 'F1 Score':
        return round(f1_score(y_true, y_pred), 6)
    elif eval_metric == 'MAPE':
        return round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 6)
    elif eval_metric == 'ROC-AUC':
        return round(roc_auc_score(y_true, y_pred), 6)
    else:
        raise ValueError(f"Error while evaluating the current model. The eval_metric should be one of the following: 'R2', 'MAE', 'MSE', 'RMSE', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'MAPE', 'ROC-AUC'. Got {eval_metric}")
        
def evaluate_model_perf(
    ml_task_type, 
    y_test,
    y_pred
) -> dict:
    """
    Evaluates how good are the predictions by comparing them with the actual values, returns regression evaluation scores

    Parameters
    ----------
    ml_task_type : str
        The type of the machine learning task. It can be either 'Regression' or 'Classification'

    y_test : np.ndarray
        The actual values of the target column.
    
    y_pred : np.ndarray
        The predicted values of the target column.
    
    Returns
    -------
    dict
        A dictionary containing the evaluation metric of the current task
            
            * R2, MAE, MSE, RMSE, MAPE for Regression tasks

            * Accuracy, Precision, Recall, F1 Score, ROC-AUC for Classification tasks
    """

    if ml_task_type == "Regression":
        r2 = _evaluate_preds(y_test, y_pred, 'R2')
        mae = _evaluate_preds(y_test, y_pred, 'MAE')
        mse = _evaluate_preds(y_test, y_pred, 'MSE')
        rmse = _evaluate_preds(y_test, y_pred, 'RMSE')
        mape = _evaluate_preds(y_test, y_pred, 'MAPE')
        return {
            "R2": r2,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape
        }
    
    elif ml_task_type == "Classification":
        accuracy = _evaluate_preds(y_test, y_pred, 'Accuracy')
        precision = _evaluate_preds(y_test, y_pred, 'Precision')
        recall = _evaluate_preds(y_test, y_pred, 'Recall')
        f1 = _evaluate_preds(y_test, y_pred, 'F1 Score')
        roc_auc = _evaluate_preds(y_test, y_pred, 'ROC-AUC')
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc
        }
    
    else:
        raise ValueError(f"Unsupported task type, only 'Regression' and 'Classification' tasks are supported, got {ml_task_type}")