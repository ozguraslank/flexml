from flexml.config.ml_models import QUICK_REGRESSION_MODELS, WIDE_REGRESSION_MODELS, QUICK_CLASSIFICATION_MODELS, WIDE_CLASSIFICATION_MODELS

# Regression & Classification ML Models
ML_MODELS = {
    "Regression": {"QUICK": QUICK_REGRESSION_MODELS,
                   "WIDE": WIDE_REGRESSION_MODELS},

    "Classification": {"QUICK": QUICK_CLASSIFICATION_MODELS,
                       "WIDE": WIDE_CLASSIFICATION_MODELS}
}

# Regression & Classification Evaluation Metrics
EVALUATION_METRICS = {
    "Regression": {"DEFAULT": "R2",
                   "ALL": ["R2", "MAE", "MSE", "RMSE"]},
                   
    "Classification": {"DEFAULT": "Accuracy",
                       "ALL": ["Accuracy", "Precision", "Recall", "F1 Score"]}
}

# Model Tuning Metric Transformations
TUNING_METRIC_TRANSFORMATIONS = {
    "Regression": {
        'R2': 'r2',
        'MAE': 'neg_mean_absolute_error',
        'MSE': 'neg_mean_squared_error',
        'RMSE': 'neg_root_mean_squared_error'
    },

    "Classification": {
        'Accuracy': 'accuracy',
        'Precision': 'precision',
        'Recall': 'recall',
        'F1 Score': 'f1_weighted'
    },

    "reverse_signed_eval_metrics": ['MAE', 'MSE', 'RMSE']
    # These metrics are used in negative form for optimization processes, so we need to reverse the sign later, e.g. from -0.42 to 0.42
}