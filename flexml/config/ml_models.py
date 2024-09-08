from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier, GradientBoostingRegressor, GradientBoostingClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

# Quick Regression Models
LINEAR_REGRESSION = LinearRegression()
XGBOOST_REGRESSION = XGBRegressor()
LIGHTGBM_REGRESSION = LGBMRegressor(verbose = -1)
CATBOOST_REGRESSION = CatBoostRegressor(allow_writing_files = False, silent=True)
DECISION_TREE_REGRESSION = DecisionTreeRegressor()
RANDOM_FOREST_REGRESSION = RandomForestRegressor()

# Wide Regression Models
SVR_REGRESSION = SVR()
KNN = KNeighborsRegressor()
ADA_BOOST = AdaBoostRegressor()
GRADIENT_BOOSTING_REGRESSION = GradientBoostingRegressor()
EXTRA_TREES_REGRESSION = ExtraTreesRegressor()

# Quick Classification Models
LOGISTIC_REGRESSION = LogisticRegression()
XGBOOST_CLASSIFIER = XGBClassifier()
LIGHTGBM_CLASSIFIER = LGBMClassifier(verbose = -1)
CATBOOST_CLASSIFIER = CatBoostClassifier(allow_writing_files = False, silent=True)
DECISION_TREE_CLASSIFIER = DecisionTreeClassifier()
RANDOM_FOREST_CLASSIFIER = RandomForestClassifier()

# Wide Classification Models
SVM_CLASSIFIER = SVC()
KNN_CLASSIFIER = KNeighborsClassifier()
ADA_BOOST_CLASSIFIER = AdaBoostClassifier()
GRADIENT_BOOSTING_CLASSIFIER = GradientBoostingClassifier()
EXTRA_TREES_CLASSIFIER = ExtraTreesClassifier()
NAIVE_BAYES_CLASSIFIER = GaussianNB()

# Quick Regression Model Configurations
QUICK_REGRESSION_MODELS = [
    {
        "name": LINEAR_REGRESSION.__class__.__name__,
        "model": LINEAR_REGRESSION,
        "tuning_param_grid": {
            'fit_intercept': [True, False]
        }
    },
    {
        "name": XGBOOST_REGRESSION.__class__.__name__,
        "model": XGBOOST_REGRESSION,
        "tuning_param_grid": {
            "n_estimators": [100, 200, 500, 1000],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.5, 0.7, 1],
            "colsample_bytree": [0.5, 0.7, 1],
            "gamma": [0, 0.1, 0.2],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0, 0.1, 0.5],
            "min_child_weight": [1, 3, 5],
            "scale_pos_weight": [1, 2, 3]
        }
    },
    {
        "name": LIGHTGBM_REGRESSION.__class__.__name__,
        "model": LIGHTGBM_REGRESSION,
        "tuning_param_grid": {
            "verbose": [-1],
            "n_estimators": [100, 200, 500, 1000],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.5, 0.7, 1],
            "colsample_bytree": [0.5, 0.7, 1],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0, 0.1, 0.5],
            "min_child_weight": [1, 3, 5],
            "num_leaves": [31, 50, 100]
        }
    },
    {
        "name": CATBOOST_REGRESSION.__class__.__name__,
        "model": CATBOOST_REGRESSION,
        "tuning_param_grid": {
            "iterations": [100, 300, 500, 1000],
            "depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "l2_leaf_reg": [0.1, 1, 3, 5, 10],
            "border_count": [32, 50, 75, 100, 150]
        }
    },
    {
        "name": DECISION_TREE_REGRESSION.__class__.__name__,
        "model": DECISION_TREE_REGRESSION,
        "tuning_param_grid": {
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10, 15],
            "min_samples_leaf": [1, 2, 4, 6],
            "max_features": ["sqrt", "log2"],
            "max_leaf_nodes": [None, 10, 20, 30, 40],
            "criterion": ["friedman_mse", "poisson", "absolute_error", "squared_error"]
        }
    },
    {
        "name": RANDOM_FOREST_REGRESSION.__class__.__name__,
        "model": RANDOM_FOREST_REGRESSION,
        "tuning_param_grid": {
            "n_estimators": [100, 200, 500, 1000],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10, 15],
            "min_samples_leaf": [1, 2, 4, 6],
            "max_features": ["sqrt", "log2"],
            "bootstrap": [True, False]
        }
    }
]

# Wide Regression Model Configurations
WIDE_REGRESSION_MODELS = QUICK_REGRESSION_MODELS + [
    {
        "name": SVR_REGRESSION.__class__.__name__,
        "model": SVR_REGRESSION,
        "tuning_param_grid": {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.01, 0.1, 1],
            "epsilon": [0.1, 0.2, 0.5, 1]
        }
    },
    {
        "name": KNN.__class__.__name__,
        "model": KNN,
        "tuning_param_grid": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [10, 20, 30, 40, 50],
            "p": [1, 2]
        }
    },
    {
        "name": ADA_BOOST.__class__.__name__,
        "model": ADA_BOOST,
        "tuning_param_grid": {
            "n_estimators": [100, 200, 500, 1000],
            "learning_rate": [0.01, 0.05, 0.1, 0.5, 1],
            "loss": ["linear", "square", "exponential"]
        }
    },
    {
        "name": GRADIENT_BOOSTING_REGRESSION.__class__.__name__,
        "model": GRADIENT_BOOSTING_REGRESSION,
        "tuning_param_grid": {
            'n_estimators': [100, 200, 500, 1000],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    {
        "name": EXTRA_TREES_REGRESSION.__class__.__name__,
        "model": EXTRA_TREES_REGRESSION,
        "tuning_param_grid": {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [None, 3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ["sqrt", "log2"],
            'bootstrap': [True, False]
        }
    }
]

# Quick Classification Model Configurations
QUICK_CLASSIFICATION_MODELS = [
    {
        "name": LOGISTIC_REGRESSION.__class__.__name__,
        "model": LOGISTIC_REGRESSION,
        "tuning_param_grid": {
            "penalty": ["l2", None],
            "C": [0.01, 0.1, 1, 10, 100],
            "max_iter": [100, 200, 300, 400, 500]
        }
    },
    {
        "name": XGBOOST_CLASSIFIER.__class__.__name__,
        "model": XGBOOST_CLASSIFIER,
        "tuning_param_grid": {
            "n_estimators": [100, 300, 500, 1000],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "subsample": [0.5, 0.7, 0.9, 1],
            "colsample_bytree": [0.5, 0.7, 0.9, 1],
            "gamma": [0, 0.1, 0.2, 0.3],
            "reg_alpha": [0, 0.1, 0.5, 1],
            "reg_lambda": [0, 0.1, 0.5, 1],
            "min_child_weight": [1, 3, 5],
            "scale_pos_weight": [1, 2, 3]
        }
    },
    {
        "name": LIGHTGBM_CLASSIFIER.__class__.__name__,
        "model": LIGHTGBM_CLASSIFIER,
        "tuning_param_grid": {
            "n_estimators": [100, 300, 500, 700],
            "max_depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "subsample": [0.5, 0.7, 0.9, 1],
            "colsample_bytree": [0.5, 0.7, 0.9, 1],
            "reg_alpha": [0, 0.1, 0.5, 1],
            "reg_lambda": [0, 0.1, 0.5, 1],
            "min_child_weight": [1, 3, 5],
            "num_leaves": [31, 50, 75, 100]
        }
    },
    {
        "name": CATBOOST_CLASSIFIER.__class__.__name__,
        "model": CATBOOST_CLASSIFIER,
        "tuning_param_grid": {
            "iterations": [100, 300, 500, 700],
            "depth": [3, 5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "l2_leaf_reg": [0.1, 1, 3, 5, 10],
            "border_count": [32, 50, 75, 100, 150]
        }
    },
    {
        "name": DECISION_TREE_CLASSIFIER.__class__.__name__,
        "model": DECISION_TREE_CLASSIFIER,
        "tuning_param_grid": {
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10, 15],
            "min_samples_leaf": [1, 2, 4, 6],
            "max_features": ["sqrt", "log2"],
            "max_leaf_nodes": [None, 10, 20, 30, 40],
            "criterion": ["gini", "entropy"]
        }
    },
    {
        "name": RANDOM_FOREST_CLASSIFIER.__class__.__name__,
        "model": RANDOM_FOREST_CLASSIFIER,
        "tuning_param_grid": {
            "n_estimators": [100, 200, 500, 1000],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10, 15],
            "min_samples_leaf": [1, 2, 4, 6],
            "max_features": ["sqrt", "log2"],
            "bootstrap": [True, False]
        }
    },
]

# Wide Classification Model Configurations
WIDE_CLASSIFICATION_MODELS = QUICK_CLASSIFICATION_MODELS + [
    {
        "name": SVM_CLASSIFIER.__class__.__name__,
        "model": SVM_CLASSIFIER,
        "tuning_param_grid": {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.01, 0.1, 1],
            "degree": [2, 3, 4, 5]
        }
    },
    {
        "name": KNN_CLASSIFIER.__class__.__name__,
        "model": KNN_CLASSIFIER,
        "tuning_param_grid": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": [10, 20, 30, 40, 50],
            "p": [1, 2]
        }
    },
    {
        "name": ADA_BOOST_CLASSIFIER.__class__.__name__,
        "model": ADA_BOOST_CLASSIFIER,
        "tuning_param_grid": {
            "n_estimators": [50, 100, 200, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.5, 1],
            "algorithm": ["SAMME", "SAMME.R"]
        }
    },
    {
        "name": GRADIENT_BOOSTING_CLASSIFIER.__class__.__name__,
        "model": GRADIENT_BOOSTING_CLASSIFIER,
        "tuning_param_grid": {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    {
        "name": EXTRA_TREES_CLASSIFIER.__class__.__name__,
        "model": EXTRA_TREES_CLASSIFIER,
        "tuning_param_grid": {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ["sqrt", "log2"],
            'bootstrap': [True, False]
        }
    },
    {
        "name": NAIVE_BAYES_CLASSIFIER.__class__.__name__,
        "model": NAIVE_BAYES_CLASSIFIER,
        "tuning_param_grid": {
            "var_smoothing": [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
        }
    }
]

