import pandas as pd
from typing import Optional, Any, Iterator
from sklearn.model_selection import (KFold, StratifiedKFold, ShuffleSplit, 
                                     StratifiedShuffleSplit, train_test_split,
                                     GroupKFold, GroupShuffleSplit)
from flexml.config import CROSS_VALIDATION_METHODS
from flexml.helpers import cross_validation_checker
from flexml.logger import get_logger


def get_cv_splits(
    df: pd.DataFrame,
    cv_method: str = "kfold",
    n_folds: Optional[int] = None,
    test_size: Optional[float] = None,
    y_array: Optional[pd.Series] = None,
    groups_col: Optional[str] = None,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    ml_task_type: Optional[str] = None,
    logging_to_file: str = False
) -> Iterator[Any]:
    """
    Returns indices for cross-validation splits according to the specified method and parameters.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset (features and target combined)

    cv_method : str, (default='kfold' for Regression, 'stratified_kfold' for Classification If `ml_task_type` is provided, else 'kfold')
        Cross-validation method to use. Options:
        - For Regression:
            - "kfold" (default) (Provide `n_folds`)
            - "holdout" (Provide `test_size`)
            - "shuffle_split" (Provide `n_folds` and `test_size`)
            - "group_kfold" (Provide `n_folds` and `groups_col`)
            - "group_shuffle_split" (Provide `n_folds`, `test_size`, and `groups_col`)
        
        - For Classification:
            - "kfold" (Provide `n_folds`)
            - "stratified_kfold" (default) (Provide `n_folds`)
            - "holdout" (Provide `test_size`)
            - "stratified_shuffle_split" (Provide `n_folds`, `test_size`)
            - "group_kfold" (Provide `n_folds` and `groups_col`)
            - "group_shuffles_plit" (Provide `n_folds`, `test_size`, and `groups_col`)

    n_folds : int, optional (default=None for hold-out validation, 5 for other cv methods)
        Number of splits/folds for methods that use folds. Default is 5

    test_size : float, optional
        The test size to use for holdout, shuffle-based methods, or group shuffle split

    y_array : pd.Series or array-like, optional
        The target variable. Required for stratified splits to ensure class balance in each fold

    groups_col : str, optional
        The name of the column in `df` that contains group labels. Required for group-based methods

    random_state : int, optional (default=None)
        The random state value for the data processing process (Ignored If 'shuffle' is set to False)

    shuffle: bool, (default=True)
        If True, the data will be shuffled before the model training process

    ml_task_type : str, optional
        The type of ML task. Options: "Regression" or "Classification"

        If you don't pass a value, the function won't accept None value for cv_method since It won't know the default cv method for your task

        If you pass a value, the default `cv_method` will be set based on the task type:
        - "Regression" => "kfold"
        - "Classification" => "stratified_kfold"

    logging_to_file : bool, optional
        Whether to log to file or not. Default is False

    Returns
    -------
    generator
        A generator that yields (train_index, test_index) for each split
    """
    logger = get_logger(__name__, "PROD", logging_to_file)
    valid_methods = CROSS_VALIDATION_METHODS.get('all')

    cv_method = cross_validation_checker(
        df=df,
        cv_method=cv_method,
        n_folds=n_folds,
        test_size=test_size,
        groups_col=groups_col,
        available_cv_methods=valid_methods,
        ml_task_type=ml_task_type
    )

    if cv_method == 'holdout' and not test_size:
        test_size = 0.25

    if cv_method == 'holdout' and test_size and n_folds:
        logger.warning(f"Both 'n_folds' and 'test_size' provided for {cv_method} validation method. Ignoring 'n_folds'")
        n_folds = None

    if cv_method == 'kfold' and test_size:
        logger.warning(f"Both 'n_folds' and 'test_size' provided for {cv_method} method. Ignoring 'test_size'")
        test_size = None

    if cv_method != 'holdout' and not n_folds:
        n_folds = 5

    if cv_method in ["stratified_kfold", "stratified_shuffle_split"] and y_array is None:
        error_msg = "`y_array` must be provided for stratified methods"
        logger.error(error_msg)
        raise ValueError(error_msg)

    groups = df[groups_col].values if groups_col else None
    if groups is not None and cv_method not in ["group_kfold", "group_shuffle_split"]:
        logger.warning(f"'groups_col' provided even though 'cv_method' is {cv_method}. Ignoring 'groups_col'")
        groups = None

    if cv_method == "kfold":
        splitter = KFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle)
        return splitter.split(df)
    
    elif cv_method == "shuffle_split":
        splitter = ShuffleSplit(n_splits=n_folds, test_size=test_size, random_state=random_state)
        return splitter.split(df)
        
    elif cv_method == "stratified_shuffle_split":
        splitter = StratifiedShuffleSplit(n_splits=n_folds, test_size=test_size, random_state=random_state)
        return splitter.split(df, y_array)

    elif cv_method == "group_shuffle_split":
        splitter = GroupShuffleSplit(n_splits=n_folds, test_size=test_size, random_state=random_state)
        return splitter.split(df, groups=groups)

    elif cv_method == "stratified_kfold":
        splitter = StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=shuffle) 
        return splitter.split(df, y_array)
    
    elif cv_method == "group_kfold":
        splitter = GroupKFold(n_splits=n_folds)
        return splitter.split(df, groups=groups)
    
    elif cv_method == "holdout":
        train_index, test_index = train_test_split(
                df.index,
                test_size=test_size,
                shuffle=shuffle,
                random_state=random_state,
                stratify=y_array if cv_method == "stratified_kfold" else None
        )
        return [(train_index, test_index)]