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


def _safe_mape(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """
    Computes the Mean Absolute Percentage Error (MAPE) while ignoring zero values in y_true since MAPE is undefined for zero values.

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        The actual values of the target column

    y_pred : pd.Series or np.ndarray
        The predicted values of the target column

    Returns
    -------
    float
        The MAPE score for the desired eval metric
    """
    mask = y_true != 0  # Ignore zero values in y_true
    return round(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])), 6)

def _evaluate_preds(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    eval_metric: str,
    average: str = 'macro'
) -> float:
    """
    Evaluates the model with the given evaluation metric by using the test set

    Parameters
    ----------
    y_true : pd.Series or np.ndarray
        The actual values of the target column

    y_pred : pd.Series or np.ndarray
        The predicted values/probabilities of the target column

    eval_metric : str
        The evaluation metric that will be used to evaluate the model   
                 
        - Avaiable evalulation metrics for Regression:    
            - R2, MAE, MSE, RMSE, MAPE

        - Avaiable evalulation metrics for Classification:    
            - Accuracy, Precision, Recall, F1 Score, ROC-AUC
        
    average : str, default='macro'
        The averaging method to use for multiclass classification metrics.
        Options are ['binary', 'micro', 'macro', 'weighted'].
        For binary classification, 'binary' is recommended.
        For multiclass, 'macro' treats all classes equally.

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
    elif eval_metric == 'MAPE':
        return _safe_mape(y_true, y_pred)
    elif eval_metric == 'Accuracy':
        return round(accuracy_score(y_true, y_pred), 6)
    elif eval_metric == 'Precision':
        return round(precision_score(y_true, y_pred, average=average), 6)
    elif eval_metric == 'Recall':
        return round(recall_score(y_true, y_pred, average=average), 6)
    elif eval_metric == 'F1 Score':
        return round(f1_score(y_true, y_pred, average=average), 6)
    elif eval_metric == 'ROC-AUC':
        if len(y_pred.shape) > 1: # If probabilites are returned 
            if y_pred.shape[1] >= 3: # If there are 3 or more classes
                return round(roc_auc_score(y_true, y_pred, average=average, multi_class='ovr'), 6)
            elif y_pred.shape[1] == 2: # If there are 2 classes
                return round(roc_auc_score(y_true, y_pred[:, 1]), 6)
        else: # If class labels are returned, ROC-AUC is not applicable (Some models don't have predict_proba method)
            return -1.0
    else:
        raise ValueError(f"Error while evaluating the current model. The eval_metric should be one of the following: 'R2', 'MAE', 'MSE', 'RMSE', 'MAPE', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'. Got {eval_metric}")
        
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
        For regression tasks: The predicted values of the target column.
        For classification tasks: The predicted probabilities for each class.
        Note: Some models like Perceptron, PassiveAggressiveClassifier, etc. don't have predict_proba method, so they return class labels directly.
    
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
    
    else: # Classification
        # Convert probabilities to class labels for metrics except ROC-AUC if y_pred is probabilities
        if len(y_pred.shape) > 1:
            y_pred_labels = np.argmax(y_pred, axis=1)
        else:
            y_pred_labels = (y_pred > 0.5).astype(int)

        # Determine appropriate averaging method based on number of classes
        n_classes = len(np.unique(y_test))
        avg_method = 'binary' if n_classes == 2 else 'macro'
        
        # Use labels for standard classification metrics
        accuracy = _evaluate_preds(y_test, y_pred_labels, 'Accuracy')
        precision = _evaluate_preds(y_test, y_pred_labels, 'Precision', average=avg_method)
        recall = _evaluate_preds(y_test, y_pred_labels, 'Recall', average=avg_method)
        f1 = _evaluate_preds(y_test, y_pred_labels, 'F1 Score', average=avg_method)
        
        # Use probabilities for ROC-AUC
        roc_auc = _evaluate_preds(y_test, y_pred, 'ROC-AUC', average=avg_method)
        
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": roc_auc
        }