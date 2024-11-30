from flexml.structures.supervised_base import SupervisedBase

class Regression(SupervisedBase):
    """
    A class to train and evaluate different regression models.

    Parameters
    ----------
    data : pd.DataFrame
        The input data for the model training process.
    
    target_col : str
        The target column name in the data.

    logging_to_file: bool, (default=False)
        If True, the logs will be saved to a file in the current path, located in /logs/flexml_logs.log, Otherwise, it will not be saved.

    Example
    -------
    >>> from flexml import Regression
    >>> df = pd.read_csv("MY_DATA.csv")
    >>> reg_exp = Regression(data=df, target_col='target_col')
    >>> reg_exp.start_experiment(experiment_size = 'quick')
    >>> reg_exp.show_model_stats(eval_metric='r2')

    ------------------------------------------------------------
    | model_name            |   r2   |   mae   | mse  |  rmse  |
    ------------------------|--------|---------|------|--------|
    | LinearRegression      | 0.7863 | 0.6721  |0.5921| 0.2469 |
    | DecisionTreeRegressor | 0.7725 | 0.6441  |0.4642| 0.4347 |
    | LGBMRegressor         | 0.7521 | 0.4751  |0.3531| 0.1445 |
    | Ridge                 | 0.7011 | 0.7590  |0.6155| 0.3411 |
    | XGBRegressor          | 0.6213 | 0.4701  |0.2923| 0.4039 |
    | DecisionTreeRegressor | 0.6096 | 0.4541  |0.2821| 0.4011 |
    | ElasticNet            | 0.5812 | 0.4201  |0.2111| 0.3011 |
    | Lasso                 | 0.5209 | 0.4101  |0.2011| 0.2911 |
    ------------------------------------------------------------
    >>> best_model = reg_exp.get_best_models(eval_metric = 'r2')
    """
    pass