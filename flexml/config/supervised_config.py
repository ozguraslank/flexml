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
    "Regression": {"DEFAULT": "r2",
                   "ALL": ["r2", "mae", "mse", "rmse"]},
                   
    "Classification": {"DEFAULT": "accuracy",
                       "ALL": ["accuracy", "precision", "recall", "f1_score"]}
}
