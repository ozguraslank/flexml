# Regression & Classification Evaluation Metrics
EVALUATION_METRICS = {
    "Regression": {"DEFAULT": "R2",
                   "ALL": ["R2", "MAE", "MSE", "RMSE", "MAPE"]},
                   
    "Classification": {"DEFAULT": "Accuracy",
                       "ALL": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]}
}

# Model Tuning Metric Transformations
TUNING_METRIC_TRANSFORMATIONS = {
    "Regression": {
        'R2': 'r2',
        'MAE': 'neg_mean_absolute_error',
        'MSE': 'neg_mean_squared_error',
        'RMSE': 'neg_root_mean_squared_error',
        'MAPE': 'neg_mean_absolute_percentage_error'
    },

    "Classification": {
        'Accuracy': 'accuracy',
        'Precision': 'precision',
        'Recall': 'recall',
        'F1 Score': 'f1_weighted',
        'ROC-AUC': 'roc_auc'
    },

    "reverse_signed_eval_metrics": ['MAE','MSE', 'RMSE', 'MAPE']
    # These metrics are used in negative form for optimization processes, so we need to reverse the sign later, e.g. from -0.42 to 0.42
}

# Supported Cross Validation Methods
CROSS_VALIDATION_METHODS = {
    # 'all' used for the get_cv_splits() function at helpers/cross_validation.py, while others are used for SupervisedBase's validations
    'all': {
        'kfold': 'kfold',
        'stratified_kfold': 'stratifiedkfold',
        'holdout': 'holdout',
        'stratified_shuffle_split': 'stratifiedshufflesplit',
        'shuffle_split': 'shufflesplit',
        'group_kfold': 'groupkfold',
        'group_shuffle_split': 'groupshufflesplit'
    },

    'Regression': {
        'kfold': 'kfold',
        'holdout': 'holdout',
        'shuffle_split': 'shufflesplit',
        'group_kfold': 'groupkfold',
        'group_shuffle_split': 'groupshufflesplit'
    },

    'Classification': {
        'kfold': 'kfold',
        'stratified_kfold': 'stratifiedkfold',
        'holdout': 'holdout',
        'shuffle_split': 'shufflesplit',
        'stratified_shuffle_split': 'stratifiedshufflesplit',
        'group_kfold': 'groupkfold',
        'group_shuffle_split': 'groupshufflesplit'
    }
}

# Feature Engineering Methods That Can Be Used
FEATURE_ENGINEERING_METHODS = {
    "accepted_numeric_imputations_methods": ['median', 'mean', 'mode', 'constant', 'drop'],
    "accepted_categorical_imputations_methods": ['mode', 'constant', 'drop'],
    "accepted_encoding_methods": ['label_encoder', 'onehot_encoder', 'ordinal_encoder'],
    "accepted_standardization_methods": ['standard_scaler', 'normalize_scaler', 'robust_scaler', 'quantile_transformer', 'minmax_scaler', 'maxabs_scaler']
}

# Supported Plot Types
PLOT_TYPES = {
    "Regression": ["feature_importance", "residuals", "prediction_error", "shap_summary","shap_violin"],
    "Classification": ["feature_importance", "confusion_matrix", "roc_curve", "shap_summary", "shap_violin", "calibration_curve"]
}