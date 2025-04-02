from flexml.structures.supervised_base import SupervisedBase

class Regression(SupervisedBase):
    """
    A class to train and evaluate different regression models

    Parameters
    ----------
    data : pd.DataFrame
        The input data for the model training process
    
    target_col : str
        The target column name in the data

    random_state : int, (default=42)
        The random state for data processing processes
    
    drop_columns : list, default=None
        Columns that will be dropped from the data
    
    categorical_imputation_method : str, default='mode'
        Imputation method for categorical columns. Options:
        * 'mode': Replace missing values with the most frequent value
        * 'constant': Replace missing values with a constant value
        * 'drop': Drop rows with missing values

    numerical_imputation_method : str, default='mean'
        Imputation method for numerical columns. Options:
        * 'mean': Replace missing values with the column mean
        * 'median': Replace missing values with the column median
        * 'mode': Replace missing values with the column mode
        * 'constant': Replace missing values with a constant value
        * 'drop': Drop rows with missing values

    column_imputation_map : dict, default=None
        Custom mapping of columns to specific imputation methods
        Example usage: {'column_name': 'mean', 'column_name2': 'mode'}

    categorical_imputation_constant : str, default='Unknown'
        The constant value for imputing categorical columns when 'constant' is selected

    numerical_imputation_constant : float, default=0.0
        The constant value for imputing numerical columns when 'constant' is selected

    encoding_method : str, default='onehot_encoder'
        Encoding method for categorical columns. Options:
        * 'label_encoder': Use label encoding
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
        * 'onehot_encoder': Use one-hot encoding
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
        * 'ordinal_encoder': Use ordinal encoding
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
        
    onehot_limit : int, default=25
        Maximum number of categories to use for one-hot encoding

    encoding_method_map : dict, default=None
        Custom mapping of columns to encoding methods
        Example usage: {'column_name': 'onehot_encoder', 'column_name2': 'label_encoder'}
    
    ordinal_encode_map : dict, default=None
        Custom mapping of columns to category order for ordinal encoding
        Example usage: {'column_name': ['low', 'medium', 'high']}
    
    normalize : str, default=None
        Standardize the data using StandardScaler. Options:
        * 'standard_scaler': Standardize the data using StandardScaler
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        * 'minmax_scaler': Scale the data using MinMaxScaler
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
        * 'robust_scaler': Scale the data using RobustScaler
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
        * 'quantile_transformer': Transform the data using QuantileTransformer
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
        * 'maxabs_scaler': Scale the data using MaxAbsScaler
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html
        * 'normalize_scaler': Normalize the data to unit length
            * https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
        
    shuffle: bool, (default=True)
        If True, the data will be shuffled before the model training process

    logging_to_file: bool, (default=False)
        If True, the logs will be saved to a file in the current path, located in /logs/flexml_logs.log, Otherwise, it will not be saved

    Example
    -------
    >>> from flexml import Regression
    >>> df = pd.read_csv("MY_DATA.csv")
    >>> reg_exp = Regression(data=df, target_col='target_col')
    >>> reg_exp.start_experiment(experiment_size = 'quick')
    >>> reg_exp.show_model_stats(eval_metric='r2')

    ---------------------------------------------------------------------
    | model_name            |   r2   |   mae   | mse  |  rmse  |  mape  |
    ------------------------|--------|---------|------|--------|--------|
    | LinearRegression      | 0.7863 | 0.6721  |0.5921| 0.2469 | 0.2011 |
    | DecisionTreeRegressor | 0.7725 | 0.6441  |0.4642| 0.4347 | 0.3011 |
    | LGBMRegressor         | 0.7521 | 0.4751  |0.3531| 0.1445 | 0.1011 |
    | Ridge                 | 0.7011 | 0.7590  |0.6155| 0.3411 | 0.2011 |
    | XGBRegressor          | 0.6213 | 0.4701  |0.2923| 0.4039 | 0.3011 |
    | DecisionTreeRegressor | 0.6096 | 0.4541  |0.2821| 0.4011 | 0.3011 |
    | ElasticNet            | 0.5812 | 0.4201  |0.2111| 0.3011 | 0.2011 |
    | Lasso                 | 0.5209 | 0.4101  |0.2011| 0.2911 | 0.2011 |
    ---------------------------------------------------------------------
    >>> best_model = reg_exp.get_best_models(eval_metric = 'r2')
    """
    pass