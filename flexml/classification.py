from flexml.structures.supervised_base import SupervisedBase

class Classification(SupervisedBase):
    """
    A class to train and evaluate different classification models.

    Parameters
    ----------
    data : pd.DataFrame
        The input data for the model training process.
    
    target_col : str
        The target column name in the data.

    logging_to_file: bool, (default=False)
        If True, the logs will be saved to a file in the current path, located in /logs/flexml_logs.log, Otherwise, it will not be saved.
    
    random_state : int, (default=42)
        The random state for data processing processes
        
    Example
    -------
    >>> from flexml import Classification
    >>> df = pd.read_csv("MY_DATA.csv")
    >>> classification_exp = Classification(data=df, target_col='target_col')
    >>> classification_exp.start_experiment(experiment_size = 'quick')
    >>> classification_exp.show_model_stats(eval_metric='accuracy')

    ------------------------------------------------------------
    | model_name            |accuracy|precision|recall|f1_score|
    ------------------------|--------|---------|------|--------|
    | LogisticRegression    | 0.7863 | 0.6721  |0.5921| 0.2469 |
    | DecisionTreeClassifier| 0.7725 | 0.6441  |0.4642| 0.4347 |
    | LGBMClassifier        | 0.7521 | 0.4751  |0.3531| 0.1445 |
    | RidgeClassifier       | 0.7011 | 0.7590  |0.6155| 0.3411 |
    | XGBClassifier         | 0.6213 | 0.4701  |0.2923| 0.4039 |
    ------------------------------------------------------------
    >>> best_model = classification_exp.get_best_models(eval_metric = 'accuracy')
    """
    pass