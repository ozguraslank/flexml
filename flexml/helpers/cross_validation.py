import pandas as pd
from typing import Optional, Any, Iterator
from sklearn.model_selection import (KFold, StratifiedKFold, ShuffleSplit, 
                                     StratifiedShuffleSplit, train_test_split,
                                     GroupKFold, GroupShuffleSplit)
from flexml.logger.logger import get_logger


def get_cv_splits(
    df: pd.DataFrame,
    cv_method: str = "k-fold",
    n_folds: Optional[int] = None,
    test_size: Optional[float] = None,
    y_label: Optional[pd.Series] = None,
    groups_col: Optional[str] = None,
    random_state: int = 42,
    logging_to_file: str = False
) -> Iterator[Any]:
    """
    Returns indices for cross-validation splits according to the specified method and parameters.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset (features and target combined)

    cv_method : str, optional
        The cross-validation method to use. Options:
        - "k-fold" (default) (Provide `n_folds` only)
        - "hold-out" (Provide `test_size` only)
        - "StratifiedKfold" (Provide `n_folds` and `y_label`)
        - "ShuffleSplit" (Provide `n_folds` and `test_size`)
        - "StratifiedShuffleSplit" (Provide `n_folds`, `test_size`, and `y_label`)
        - "GroupKFold" (Provide `n_folds` and `groups_col`)
        - "GroupShuffleSplit" (Provide `n_folds`, `test_size`, and `groups_col`)

    n_folds : int, optional
        Number of splits/folds for methods that use folds. Default is 5

    test_size : float, optional
        The test size to use for hold-out, shuffle-based methods, or group shuffle split

    y_label : pd.Series or array-like, optional
        The target variable. Required for stratified splits to ensure class balance in each fold

    groups_col : str, optional
        The name of the column in `df` that contains group labels. Required for group-based methods

    random_state : int, (default=42)
        Random seed value

    logging_to_file : bool, optional
        Whether to log to file or not. Default is False

    Returns
    -------
    generator
        A generator that yields (train_index, test_index) for each split
    """
    logger = get_logger(__name__, "PROD", logging_to_file)

    valid_methods = [
        "k-fold", "hold-out", "StratifiedKfold", "ShuffleSplit", 
        "StratifiedShuffleSplit", "GroupKFold", "GroupShuffleSplit"
    ]

    if cv_method not in valid_methods:
        error_msg = f"`cv_method` must be one of {valid_methods}, got '{cv_method}'"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if n_folds is not None and (not isinstance(n_folds, int) or n_folds < 2):
        error_msg = "`n_folds` must be an integer >= 2 if provided"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if test_size is not None and (not 0 < test_size < 1):
        error_msg = "`test_size` must be a float between 0 and 1 if provided"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if groups_col is not None and groups_col not in df.columns:
        error_msg = f"`groups_col` must be a column in `df`, got '{groups_col}'"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if cv_method in ["StratifiedKfold", "StratifiedShuffleSplit"] and y_label is None:
        error_msg = "`y_label` must be provided for stratified methods"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if cv_method in ["GroupKFold", "GroupShuffleSplit"] and groups_col is None:
        error_msg = "`groups_col` must be provided for group-based methods"
        logger.error(error_msg)
        raise ValueError(error_msg)

    if cv_method == 'hold-out' and not test_size:
        logger.warning("No 'test_size' provided for hold-out method. Defaulting to 0.25")
        test_size = 0.25

    if cv_method == 'hold-out' and test_size and n_folds:
        logger.warning("Both 'n_folds' and 'test_size' provided. Ignoring 'n_folds'")
        n_folds = None

    if cv_method == 'k-fold' and test_size:
        logger.warning(f"Both 'n_folds' and 'test_size' provided for {cv_method} method. Ignoring 'test_size'")
        test_size = None

    if cv_method != 'hold-out' and not n_folds:
        n_folds = 5

    groups = df[groups_col].values if groups_col else None

    if groups is not None and cv_method not in ["GroupKFold", "GroupShuffleSplit"]:
        logger.warning(f"'groups_col' provided even though 'cv_method' is {cv_method}. Ignoring 'groups_col'")
        groups = None

    # Splitter selection
    if n_folds and test_size:
        # Shuffle-based methods
        if cv_method == "StratifiedShuffleSplit":
            splitter = StratifiedShuffleSplit(n_splits=n_folds, test_size=test_size, random_state=random_state)
            return splitter.split(df, y_label)
        
        elif cv_method == "GroupShuffleSplit":
            splitter = GroupShuffleSplit(n_splits=n_folds, test_size=test_size, random_state=random_state)
            return splitter.split(df, groups=groups)
        
        else:
            splitter = ShuffleSplit(n_splits=n_folds, test_size=test_size, random_state=random_state)
            return splitter.split(df)

    if n_folds and not test_size:
        # Fold-based methods
        if cv_method == "StratifiedKfold":
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            return splitter.split(df, y_label)
        
        elif cv_method == "GroupKFold":
            splitter = GroupKFold(n_splits=n_folds)
            return splitter.split(df, groups=groups)
        
        else:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            return splitter.split(df)

    if test_size and not n_folds:
        # Single split methods
        if cv_method == "GroupShuffleSplit":
            splitter = GroupShuffleSplit(n_folds=1, test_size=test_size, random_state=random_state)
            return splitter.split(df, groups=groups)
        
        else:
            train_index, test_index = train_test_split(
                df.index,
                test_size=test_size,
                shuffle=True,
                random_state=random_state,
                stratify=y_label if cv_method == "StratifiedKfold" else None
            )
            return [(train_index, test_index)]

    # Default fallback
    splitter = KFold(n_folds=5, shuffle=True, random_state=random_state)
    return splitter.split(df)