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

    experiment_size : str, (default='quick')
        The size of the experiment to run. It can be 'quick' or 'wide'
        
        * If It's selected 'quick', quick amount of machine learning models will be used to get quick results
        * If It's selected 'wide', wide range of machine learning models will be used to get more comprehensive results
    
    test_size : float, default=0.25
        The size of the test data in the train-test split process.
    
    random_state : int, default=42
        The random state value for the train-test split process
        For more info, visit https://scikit-learn.org/stable/glossary.html#term-random_state

    logging_to_file: bool, (default=False)
        If True, the logs will be saved to a file in the current path, located in /logs/flexml_logs.log, Otherwise, it will not be saved.

    Example
    -------
    >>> from flexml.classification import Classification
    >>> df = pd.read_csv("MY_DATA.csv")
    >>> classification_exp = Classification(data=df, target_col='target_col', experiment_size='quick')
    >>> classification_exp.start_experiment()
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