from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# TODO: Should be improved
def get_ml_models(
    ml_task_type: str,
    num_class: Optional[int] = None,
    random_state: Optional[int] = None,
    n_jobs: Optional[int] = -1
) -> dict:
    """
    Returns a dictionary of quick and wide regression and classification models

    Parameters
    ----------
    ml_task_type : str
        The type of the machine learning task. It can be "Regression" or "Classification"

    num_class : int, optional (default=None)
        The number of classes in the classification task. No need to pass it in regression tasks
        It will be set to 2 if None is passed to suppose its binary classification

    random_state : int, optional (default=None)
        The random state value for the model training process

    n_jobs : int, optional (default=-1)
        The number of jobs to run in parallel. -1 means using all processors
    
    Returns
    -------
    dict
        A dictionary of quick and wide Regression/Classification models
    """
    if ml_task_type not in ["Regression", "Classification"]:
        raise ValueError(f"Expected ml_task_type to be either 'Regression' or 'Classification', got {ml_task_type}")

    if ml_task_type == "Regression":
        from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, HuberRegressor
        from sklearn.linear_model import BayesianRidge, OrthogonalMatchingPursuit
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import (
            AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, 
            ExtraTreesRegressor, HistGradientBoostingRegressor
        )
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.neural_network import MLPRegressor
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor


        # Quick Regression Models
        LINEAR_REGRESSION = LinearRegression(n_jobs=n_jobs)
        LASSO_REGRESSION = Lasso(random_state=random_state)
        RIDGE_REGRESSION = Ridge(random_state=random_state)
        XGBOOST_REGRESSION = XGBRegressor(enable_categorical=True, random_state=random_state, n_jobs=n_jobs)
        LIGHTGBM_REGRESSION = LGBMRegressor(verbose=-1, enable_categorical=True, random_state=random_state, n_jobs=n_jobs)
        CATBOOST_REGRESSION = CatBoostRegressor(allow_writing_files=False, silent=True, random_seed=random_state, thread_count=n_jobs)
        DECISION_TREE_REGRESSION = DecisionTreeRegressor(random_state=random_state)
        ELASTIC_NET_REGRESSION = ElasticNet(random_state=random_state)
        HUBER_REGRESSION = HuberRegressor()

        # Wide Regression Models
        KNN_REGRESSION = KNeighborsRegressor(n_jobs=n_jobs) 
        BAYESIAN_RIDGE_REGRESSION = BayesianRidge()
        ADA_BOOST_REGRESSION = AdaBoostRegressor(random_state=random_state)
        HIST_GRADIENT_BOOSTING_REGRESSION = HistGradientBoostingRegressor(random_state=random_state)
        GRADIENT_BOOSTING_REGRESSION = GradientBoostingRegressor(random_state=random_state)
        RANDOM_FOREST_REGRESSION = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
        EXTRA_TREES_REGRESSION = ExtraTreesRegressor(random_state=random_state, n_jobs=n_jobs)
        OMP_REGRESSION = OrthogonalMatchingPursuit()
        MLP_REGRESSION = MLPRegressor(
            solver='lbfgs',
            hidden_layer_sizes=(50,),
            early_stopping=True,
            learning_rate='adaptive',
            random_state=random_state
        )

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
                "name": LASSO_REGRESSION.__class__.__name__,
                "model": LASSO_REGRESSION,
                "tuning_param_grid": {
                    "alpha": [0.1, 0.5, 1.0, 2.0],
                    "max_iter": [1000, 2000, 3000]
                }
            },
            {
                "name": RIDGE_REGRESSION.__class__.__name__,
                "model": RIDGE_REGRESSION,
                "tuning_param_grid": {
                    "alpha": [0.1, 0.5, 1.0, 2.0],
                    "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
                }
            },
            {
                "name": XGBOOST_REGRESSION.__class__.__name__,
                "model": XGBOOST_REGRESSION,
                "tuning_param_grid": {
                    "n_estimators": [100, 200, 300, 500, 700, 1000],
                    "max_depth": [3, 5, 7, 9, 10],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
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
                    "n_estimators": [100, 200, 300, 500, 700, 1000],
                    "max_depth": [3, 5, 7, 9, 10, 12],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
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
                    "iterations": [100, 200, 300, 500, 700, 1000, 1500],
                    "depth": [3, 5, 7, 9, 10, 12],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                    "l2_leaf_reg": [0.1, 1, 3, 5, 10],
                    "border_count": [32, 50, 75, 100, 150]
                }
            },
            {
                "name": DECISION_TREE_REGRESSION.__class__.__name__,
                "model": DECISION_TREE_REGRESSION,
                "tuning_param_grid": {
                    "max_depth": [3, 5, 7, 9, 10],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"],
                    "max_leaf_nodes": [10, 20, 30, 40],
                    "criterion": ["friedman_mse", "poisson", "absolute_error", "squared_error"]
                }
            },
            {
                "name": ELASTIC_NET_REGRESSION.__class__.__name__,
                "model": ELASTIC_NET_REGRESSION,
                "tuning_param_grid": {
                    "alpha": [0.1, 0.5, 1.0, 2.0],
                    "l1_ratio": [0.1, 0.5, 0.7, 1.0]
                }
            },
            {
                "name": HUBER_REGRESSION.__class__.__name__,
                "model": HUBER_REGRESSION,
                "tuning_param_grid": {
                    "epsilon": [1.1, 1.35, 1.5, 1.75, 2.0],
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]
                }
            }
        ]

        # Wide Regression Model Configurations
        WIDE_REGRESSION_MODELS = QUICK_REGRESSION_MODELS + [
            {
                "name": KNN_REGRESSION.__class__.__name__,
                "model": KNN_REGRESSION,
                "tuning_param_grid": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2]
                }
            },
            {
                "name": ADA_BOOST_REGRESSION.__class__.__name__,
                "model": ADA_BOOST_REGRESSION,
                "tuning_param_grid": {
                    "n_estimators": [50, 100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1],
                    "loss": ["linear", "square", "exponential"]
                }
            },
            {
                "name": BAYESIAN_RIDGE_REGRESSION.__class__.__name__,
                "model": BAYESIAN_RIDGE_REGRESSION,
                "tuning_param_grid": {
                    "max_iter": [100, 200, 300, 400, 500],
                    "alpha_1": [1e-6, 1e-5, 1e-4],
                    "alpha_2": [1e-6, 1e-5, 1e-4],
                    "lambda_1": [1e-6, 1e-5, 1e-4],
                    "lambda_2": [1e-6, 1e-5, 1e-4]
                }
            },
            {
                "name": RANDOM_FOREST_REGRESSION.__class__.__name__,
                "model": RANDOM_FOREST_REGRESSION,
                "tuning_param_grid": {
                    "n_estimators": [50, 100, 200, 300, 400],
                    "max_depth": [3, 5, 7, 9, 10],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", 0.3, 0.5],
                    "bootstrap": [True, False]
                }
            },
            {
                "name": EXTRA_TREES_REGRESSION.__class__.__name__,
                "model": EXTRA_TREES_REGRESSION,
                "tuning_param_grid": {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [3, 5, 7, 9, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ["sqrt", "log2"],
                    'bootstrap': [True, False]
                }
            },
            {
                "name": OMP_REGRESSION.__class__.__name__,
                "model": OMP_REGRESSION,
                "tuning_param_grid": {
                    "n_nonzero_coefs": [5, 10, 15, 20],
                    "tol": [1e-4, 1e-3, 1e-2, 1e-1]
                }
            },
            {
                "name": HIST_GRADIENT_BOOSTING_REGRESSION.__class__.__name__,
                "model": HIST_GRADIENT_BOOSTING_REGRESSION,
                "tuning_param_grid": {
                    "max_iter": [100, 200, 300, 500],
                    "max_depth": [3, 5, 7, 9, 10],
                    "learning_rate": [0.01, 0.1, 0.3],
                    "min_samples_leaf": [1, 5, 10],
                    "l2_regularization": [0, 1.0, 10.0],
                    "max_bins": [128, 255]
                }
            },
            {
                "name": GRADIENT_BOOSTING_REGRESSION.__class__.__name__,
                "model": GRADIENT_BOOSTING_REGRESSION,
                "tuning_param_grid": {
                    "n_estimators": [100, 200, 300, 400, 500],
                    "max_depth": [3, 5, 7, 9, 10],
                    "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "alpha": [0.1, 0.5, 0.9]
                }
            },
            {
                "name": MLP_REGRESSION.__class__.__name__,
                "model": MLP_REGRESSION,
                "tuning_param_grid": {
                    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                    "max_iter": [100, 200, 300, 400],
                    "activation": ["relu", "tanh"],
                    "alpha": [0.0001, 0.001, 0.01],
                    "learning_rate": ["constant", "adaptive"],
                    "learning_rate_init": [0.001, 0.01]
                }
            }
        ]

        return {
            "QUICK": QUICK_REGRESSION_MODELS,
            "WIDE": WIDE_REGRESSION_MODELS
        }

    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import (
            AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, 
            ExtraTreesClassifier, HistGradientBoostingClassifier
        )
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
        from sklearn.neural_network import MLPClassifier
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
    

        if num_class is None: # Suppose binary 
            num_class = 2

        if num_class > 2:
            xgb_objective = "multi:softmax"
        else:
            xgb_objective = "binary:logistic"

        # Quick Classification Models
        LOGISTIC_REGRESSION = LogisticRegression(max_iter=1000, random_state=random_state, n_jobs=n_jobs)
        XGBOOST_CLASSIFIER = XGBClassifier(objective=xgb_objective, random_state=random_state, n_jobs=n_jobs)
        LIGHTGBM_CLASSIFIER = LGBMClassifier(verbose=-1, random_state=random_state, n_jobs=n_jobs)
        CATBOOST_CLASSIFIER = CatBoostClassifier(allow_writing_files=False, silent=True, random_seed=random_state, thread_count=n_jobs)
        DECISION_TREE_CLASSIFIER = DecisionTreeClassifier(random_state=random_state)
        RANDOM_FOREST_CLASSIFIER = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
        NAIVE_BAYES_CLASSIFIER = GaussianNB()
        KNN_CLASSIFIER = KNeighborsClassifier(n_jobs=n_jobs)

        # Wide Classification Models
        ADA_BOOST_CLASSIFIER = AdaBoostClassifier(random_state=random_state)
        HIST_GRADIENT_BOOSTING_CLASSIFIER = HistGradientBoostingClassifier(random_state=random_state)
        GRADIENT_BOOSTING_CLASSIFIER = GradientBoostingClassifier(random_state=random_state)
        EXTRA_TREES_CLASSIFIER = ExtraTreesClassifier(random_state=random_state, n_jobs=n_jobs)
        QDA_CLASSIFIER = QuadraticDiscriminantAnalysis()
        LDA_CLASSIFIER = LinearDiscriminantAnalysis()
        MLP_CLASSIFIER = MLPClassifier(
            hidden_layer_sizes=(100,),
            early_stopping=True,
            tol=0.001,
            learning_rate='adaptive',
            random_state=random_state
        )

        # Quick Classification Model Configurations
        QUICK_CLASSIFICATION_MODELS = [
            {
                "name": LOGISTIC_REGRESSION.__class__.__name__,
                "model": LOGISTIC_REGRESSION,
                "tuning_param_grid": {
                    "penalty": ["l2"],
                    "C": [0.01, 0.1, 1, 10, 100],
                    "max_iter": [100, 200, 300, 400, 500]
                }
            },
            {
                "name": XGBOOST_CLASSIFIER.__class__.__name__,
                "model": XGBOOST_CLASSIFIER,
                "tuning_param_grid": {
                    "n_estimators": [100, 200, 300, 500, 700, 1000],
                    "max_depth": [3, 5, 7, 9, 10],
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
                    "n_estimators": [100, 200, 300, 500, 700, 1000],
                    "max_depth": [3, 5, 7, 9, 10, 12],
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
                    "iterations": [100, 200, 300, 500, 700, 1000],
                    "depth": [3, 5, 7, 9, 10, 12],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                    "l2_leaf_reg": [0.1, 1, 3, 5, 10],
                    "border_count": [32, 50, 75, 100, 150]
                }
            },
            {
                "name": DECISION_TREE_CLASSIFIER.__class__.__name__,
                "model": DECISION_TREE_CLASSIFIER,
                "tuning_param_grid": {
                    "max_depth": [3, 5, 7, 9, 10],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"],
                    "max_leaf_nodes": [10, 20, 30, 40],
                    "criterion": ["gini", "entropy"]
                }
            },
            {
                "name": RANDOM_FOREST_CLASSIFIER.__class__.__name__,
                "model": RANDOM_FOREST_CLASSIFIER,
                "tuning_param_grid": {
                    "n_estimators": [100, 200, 300, 400],
                    "max_depth": [3, 5, 7, 9, 10],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", 0.3, 0.5],
                    "bootstrap": [True, False]
                }
            },
            {
                "name": NAIVE_BAYES_CLASSIFIER.__class__.__name__,
                "model": NAIVE_BAYES_CLASSIFIER,
                "tuning_param_grid": {
                    "var_smoothing": [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
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
        ]

        # Wide Classification Model Configurations
        WIDE_CLASSIFICATION_MODELS = QUICK_CLASSIFICATION_MODELS + [
            {
                "name": ADA_BOOST_CLASSIFIER.__class__.__name__,
                "model": ADA_BOOST_CLASSIFIER,
                "tuning_param_grid": {
                    "n_estimators": [50, 100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1],
                    "algorithm": ["SAMME", "SAMME.R"]
                }
            },
            {
                "name": HIST_GRADIENT_BOOSTING_CLASSIFIER.__class__.__name__,
                "model": HIST_GRADIENT_BOOSTING_CLASSIFIER,
                "tuning_param_grid": {
                    "max_iter": [100, 200, 300, 500],
                    "max_depth": [3, 5, 7, 9, 10],
                    "learning_rate": [0.01, 0.1, 0.3],
                    "min_samples_leaf": [1, 5, 10],
                    "l2_regularization": [0, 1.0, 10.0],
                    "max_bins": [128, 255]
                }
            },
            {
                "name": GRADIENT_BOOSTING_CLASSIFIER.__class__.__name__,
                "model": GRADIENT_BOOSTING_CLASSIFIER,
                "tuning_param_grid": {
                    'n_estimators': [100, 200, 300, 500],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 5, 7, 9, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            {
                "name": EXTRA_TREES_CLASSIFIER.__class__.__name__,
                "model": EXTRA_TREES_CLASSIFIER,
                "tuning_param_grid": {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [3, 5, 7, 9, 10],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ["sqrt", "log2"],
                    'bootstrap': [True, False]
                }
            },
            {
                "name": QDA_CLASSIFIER.__class__.__name__,
                "model": QDA_CLASSIFIER,
                "tuning_param_grid": {
                    "reg_param": [0.0, 0.1, 0.5, 1.0],
                    "tol": [1e-4, 1e-3, 1e-2, 1e-1]
                }
            },
            {
                "name": LDA_CLASSIFIER.__class__.__name__,
                "model": LDA_CLASSIFIER,
                "tuning_param_grid": {
                    "solver": ["svd", "lsqr", "eigen"],
                    "shrinkage": [0.1, 0.5, 1.0]
                }
            },
            {
                "name": MLP_CLASSIFIER.__class__.__name__,
                "model": MLP_CLASSIFIER,
                "tuning_param_grid": {
                    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
                    "max_iter": [100, 200, 300, 400],
                    "activation": ["relu", "tanh"],
                    "alpha": [0.0001, 0.001, 0.01],
                    "learning_rate": ["constant", "adaptive"],
                    "learning_rate_init": [0.001, 0.01]
                }
            }
        ]

        return {
            "QUICK": QUICK_CLASSIFICATION_MODELS,
            "WIDE": WIDE_CLASSIFICATION_MODELS
        }