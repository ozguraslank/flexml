# LinearRegression, Ridge, Lasso, XGBoost, LightGBM, CatBoost
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostRegressor, AdaBoostClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

# Quick Regression Models
LINEAR_REGRESSION = LinearRegression()
RIDGE = Ridge()
LASSO = Lasso()
ELASTIC_NET = ElasticNet()
XGBOOST_REGRESSION = XGBRegressor()
LIGHTGBM_REGRESSION = LGBMRegressor(verbose = -1)
CATBOOST_REGRESSION = CatBoostRegressor(verbose = 0, allow_writing_files = False)
DECISION_TREE_REGRESSION = DecisionTreeRegressor()

# Wide Regression Models
RANDOM_FOREST_REGRESSION = RandomForestRegressor()
SVR_REGRESSION = SVR()
KNN = KNeighborsRegressor()
ADA_BOOST = AdaBoostRegressor()

# Quick Classification Models
LOGISTIC_REGRESSION = LogisticRegression()
RIDGE_CLASSIFIER = RidgeClassifier()
LASSO_CLASSIFIER = Lasso()
ELASTIC_NET_CLASSIFIER = ElasticNet()
XGBOOST_CLASSIFIER = XGBClassifier()
LIGHTGBM_CLASSIFIER = LGBMClassifier(verbose = -1)
CATBOOST_CLASSIFIER = CatBoostClassifier(verbose = 0, allow_writing_files = False)
DECISION_TREE_CLASSIFIER = DecisionTreeClassifier()

# Wide Classification Models
RANDOM_FOREST_CLASSIFIER = RandomForestClassifier()
SVM_CLASSIFIER = SVC()
KNN_CLASSIFIER = KNeighborsClassifier()
ADA_BOOST_CLASSIFIER = AdaBoostClassifier()

# Quick Regression Model Configurations
QUICK_REGRESSION_MODELS = [
    {
        "name": LINEAR_REGRESSION.__class__.__name__,
        "model": LINEAR_REGRESSION,
        "tuning_param_grid": {
            'fit_intercept': [True, False],
            'normalize': [True, False]
        }
    },
    {
        "name": RIDGE.__class__.__name__,
        "model": RIDGE,
        "tuning_param_grid": {
            "alpha": [0.1, 1, 10],
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
    },
    {
        "name": LASSO.__class__.__name__,
        "model": LASSO,
        "tuning_param_grid": {
            "alpha": [0.1, 1, 10],
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "selection": ['cyclic', 'random']
        }
    },
    {
        "name": ELASTIC_NET.__class__.__name__,
        "model": ELASTIC_NET,
        "tuning_param_grid": {
            "alpha": [0.1, 1, 10],
            "l1_ratio": [0.1, 0.5, 0.9],
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "selection": ['cyclic', 'random']
        }
    },
    {
        "name": XGBOOST_REGRESSION.__class__.__name__,
        "model": XGBOOST_REGRESSION,
        "tuning_param_grid": {
            "n_estimators": [100, 500],
            "max_depth": [3, 5, 10],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.5, 0.7, 1],
            "colsample_bytree": [0.5, 0.7, 1],
            "gamma": [0, 0.1, 0.2],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0, 0.1, 0.5]
        }
    },
    {
        "name": LIGHTGBM_REGRESSION.__class__.__name__,
        "model": LIGHTGBM_REGRESSION,
        "tuning_param_grid": {
            "verbose": [-1],
            "n_estimators": [100, 500],
            "max_depth": [3, 5, 10],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.5, 0.7, 1],
            "colsample_bytree": [0.5, 0.7, 1],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0, 0.1, 0.5]
        }
    },
    {
        "name": CATBOOST_REGRESSION.__class__.__name__,
        "model": CATBOOST_REGRESSION,
        "tuning_param_grid": {
            "allow_writing_files": [False],
            "n_estimators": [100, 500],
            "learning_rate": [0.01, 0.1, 0.3],
            "depth": [3, 5, 10],
            "l2_leaf_reg": [1, 3, 5],
            "border_count": [32, 64, 128]
        }
    },
    {
        "name": DECISION_TREE_REGRESSION.__class__.__name__,
        "model": DECISION_TREE_REGRESSION,
        "tuning_param_grid": {
            "criterion": ['mse', 'mae'],
            "splitter": ['best', 'random'],
            "max_depth": [None, 10, 50, 100],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ['auto', 'sqrt', 'log2']
        }
    }
]

# Wide Regression Model Configurations
WIDE_REGRESSION_MODELS = QUICK_REGRESSION_MODELS + [
    {
        "name": RANDOM_FOREST_REGRESSION.__class__.__name__,
        "model": RANDOM_FOREST_REGRESSION,
        "tuning_param_grid": {
            "n_estimators": [100, 500],
            "criterion": ['mse', 'mae'],
            "max_depth": [None, 10, 50, 100],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ['auto', 'sqrt', 'log2']
        }
    },
    {
        "name": SVR_REGRESSION.__class__.__name__,
        "model": SVR_REGRESSION,
        "tuning_param_grid": {
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            "degree": [3, 4, 5],
            "gamma": ['scale', 'auto'],
            "C": [0.1, 1, 10],
            "epsilon": [0.1, 0.2, 0.5]
        }
    },
    {
        "name": KNN.__class__.__name__,
        "model": KNN,
        "tuning_param_grid": {
            "n_neighbors": [3, 5, 10],
            "weights": ['uniform', 'distance'],
            "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
            "leaf_size": [30, 60, 90],
            "p": [1, 2]
        }
    },
    {
        "name": ADA_BOOST.__class__.__name__,
        "model": ADA_BOOST,
        "tuning_param_grid": {
            "n_estimators": [50, 100, 500],
            "learning_rate": [0.01, 0.1, 1],
            "loss": ['linear', 'square', 'exponential']
        }
    }
]

# Quick Classification Model Configurations
QUICK_CLASSIFICATION_MODELS = [
    {
        "name": LOGISTIC_REGRESSION.__class__.__name__,
        "model": LOGISTIC_REGRESSION,
        "tuning_param_grid": {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.1, 1, 10],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [100, 1000, 10000]
        }
    },
    {
        "name": RIDGE_CLASSIFIER.__class__.__name__,
        "model": RIDGE_CLASSIFIER,
        "tuning_param_grid": {
            "alpha": [0.1, 1, 10],
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
    },
    {
        "name": LASSO_CLASSIFIER.__class__.__name__,
        "model": LASSO_CLASSIFIER,
        "tuning_param_grid": {
            "alpha": [0.1, 1, 10],
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "selection": ['cyclic', 'random']
        }
    },
    {
        "name": ELASTIC_NET_CLASSIFIER.__class__.__name__,
        "model": ELASTIC_NET_CLASSIFIER,
        "tuning_param_grid": {
            "alpha": [0.1, 1, 10],
            "l1_ratio": [0.1, 0.5, 0.9],
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "selection": ['cyclic', 'random']
        }
    },
    {
        "name": XGBOOST_CLASSIFIER.__class__.__name__,
        "model": XGBOOST_CLASSIFIER,
        "tuning_param_grid": {
            "n_estimators": [100, 500],
            "max_depth": [3, 5, 10],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.5, 0.7, 1],
            "colsample_bytree": [0.5, 0.7, 1],
            "gamma": [0, 0.1, 0.2],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0, 0.1, 0.5]
        }
    },
    {
        "name": LIGHTGBM_CLASSIFIER.__class__.__name__,
        "model": LIGHTGBM_CLASSIFIER,
        "tuning_param_grid": {
            "verbose": [-1],
            "n_estimators": [100, 500],
            "max_depth": [3, 5, 10],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.5, 0.7, 1],
            "colsample_bytree": [0.5, 0.7, 1],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0, 0.1, 0.5]
        }
    },
    {
        "name": CATBOOST_CLASSIFIER.__class__.__name__,
        "model": CATBOOST_CLASSIFIER,
        "tuning_param_grid": {
            "allow_writing_files": [False],
            "n_estimators": [100, 500],
            "learning_rate": [0.01, 0.1, 0.3],
            "depth": [3, 5, 10],
            "l2_leaf_reg": [1, 3, 5],
            "border_count": [32, 64, 128]
        }
    },
    {
        "name": DECISION_TREE_CLASSIFIER.__class__.__name__,
        "model": DECISION_TREE_CLASSIFIER,
        "tuning_param_grid": {
            "criterion": ['gini', 'entropy'],
            "splitter": ['best', 'random'],
            "max_depth": [None, 10, 50, 100],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ['auto', 'sqrt', 'log2']
        }
    }
]

# Wide Classification Model Configurations
WIDE_CLASSIFICATION_MODELS = QUICK_CLASSIFICATION_MODELS + [
    {
        "name": RANDOM_FOREST_CLASSIFIER.__class__.__name__,
        "model": RANDOM_FOREST_CLASSIFIER,
        "tuning_param_grid": {
            "n_estimators": [100, 500],
            "criterion": ['gini', 'entropy'],
            "max_depth": [None, 10, 50, 100],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ['auto', 'sqrt', 'log2']
        }
    },
    {
        "name": SVM_CLASSIFIER.__class__.__name__,
        "model": SVM_CLASSIFIER,
        "tuning_param_grid": {
            "C": [0.1, 1, 10],
            "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            "degree": [3, 4, 5],
            "gamma": ['scale', 'auto'],
            "probability": [True]
        }
    },
    {
        "name": KNN_CLASSIFIER.__class__.__name__,
        "model": KNN_CLASSIFIER,
        "tuning_param_grid": {
            "n_neighbors": [3, 5, 10],
            "weights": ['uniform', 'distance'],
            "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
            "leaf_size": [30, 60, 90],
            "p": [1, 2]
        }
    },
    {
        "name": ADA_BOOST_CLASSIFIER.__class__.__name__,
        "model": ADA_BOOST_CLASSIFIER,
        "tuning_param_grid": {
            "n_estimators": [50, 100, 500],
            "learning_rate": [0.01, 0.1, 1],
            "algorithm": ['SAMME', 'SAMME.R']
        }
    }
]

