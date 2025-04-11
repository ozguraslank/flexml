import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, MaxAbsScaler, normalize 
from typing import List, Optional, Dict, Any
from flexml.logger import get_logger

class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    A transformer to drop specified columns from a dataset
    """
    def __init__(self, drop_columns: Optional[List[str]] = None):
        self.drop_columns = drop_columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Drops specified columns from the input DataFrame
        
        Returns
        -------
        pd.DataFrame
            A DataFrame with the specified columns dropped
        """
        return X.drop(columns=self.drop_columns, axis=1, errors='ignore')
    

class ColumnImputer(BaseEstimator, TransformerMixin):
    """
    A transformer to impute missing values in a dataset
    """
    def __init__(
        self, 
        column_imputation_mapper: Dict[str, str],
        numerical_imputation_constant: float = 0.0, 
        categorical_imputation_constant: str = "Unknown"
    ):
        self.column_imputation_mapper = column_imputation_mapper
        self.numerical_imputation_constant = numerical_imputation_constant
        self.categorical_imputation_constant = categorical_imputation_constant

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> pd.DataFrame:
        # Categorical columns are converted to string
        categorical_cols = X.select_dtypes(exclude=['number']).columns
        X[categorical_cols] = X[categorical_cols].astype(str)

        for column, method in self.column_imputation_mapper.items():
            X[column] = X[column].replace("nan", pd.NA)
            if method == "mean":
                mean_value = X[column].mean()
                X[column] = X[column].fillna(mean_value)

            elif method == "median":
                median_value = X[column].median()
                X[column] = X[column].fillna(median_value)

            elif method == "mode":
                mode_values = X[column].mode()
                if len(mode_values) > 0:
                    mode_value = mode_values[0]
                else:
                    # TODO: Notify user that mode is not available
                    mode_value = self.categorical_imputation_constant
                X[column] = X[column].replace("nan", np.nan).fillna(mode_value)

            elif method == "constant":
                if X[column].dtype != 'object':
                    constant = self.numerical_imputation_constant
                else:
                    constant = self.categorical_imputation_constant
                X[column] = X[column].replace("nan", np.nan).fillna(constant)

            elif method == "drop":
                X = X.dropna(subset=[column])

            else:
                raise ValueError(f"Invalid imputation method: {method}")
            
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    A transformer to encode categorical columns in a dataset
    """
    def __init__(
        self, 
        encoding_method_mapper: Dict[str, str], 
        ordinal_map: Dict[str, List[str]],
        onehot_limit: int = 25
    ):
        self.encoding_method_mapper = encoding_method_mapper
        self.ordinal_map = ordinal_map
        self.onehot_limit = onehot_limit
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.ordinal_encoders = {}

    def fit(self, X, y=None):
        # Categorical columns are converted to string
        categorical_cols = X.select_dtypes(exclude=['number']).columns
        X[categorical_cols] = X[categorical_cols].astype(str)

        for col, method in self.encoding_method_mapper.items():
            if method == "label_encoder":
                encoder = LabelEncoder()
                encoder.fit(X[col].fillna("Unknown"))
                self.label_encoders[col] = encoder

            elif method == "onehot_encoder":
                encoder = OneHotEncoder(
                    sparse_output=False, 
                    handle_unknown="ignore", 
                    max_categories=self.onehot_limit
                )
                encoder.fit(X[[col]])
                self.onehot_encoders[col] = encoder

            elif method == "ordinal_encoder":
                if col in self.ordinal_map:
                    categories = [self.ordinal_map[col]]
                    encoder = OrdinalEncoder(categories=categories)
                    encoder.fit(X[[col]])
                    self.ordinal_encoders[col] = encoder

        return self

    def transform(self, X) -> pd.DataFrame:
        # Categorical columns are converted to string
        categorical_cols = X.select_dtypes(exclude=['number']).columns
        X[categorical_cols] = X[categorical_cols].astype(str)

        for col, method in self.encoding_method_mapper.items():
            if method == "label_encoder":
                if col in self.label_encoders:
                    encoder = self.label_encoders[col]
                    # Identify known and unknown labels
                    known_mask = X[col].isin(encoder.classes_)
                    # Transform known labels
                    if known_mask.any():
                         X.loc[known_mask, col] = encoder.transform(X.loc[known_mask, col])
                    # Handle unknown labels
                    X.loc[~known_mask, col] = -1
                    X[col] = X[col].astype(int)

            elif method == "onehot_encoder":
                if col in self.onehot_encoders:
                    encoder = self.onehot_encoders[col]
                    one_hot_encoded = encoder.transform(X[[col]])
                    one_hot_df = pd.DataFrame(
                        one_hot_encoded,
                        columns=encoder.get_feature_names_out([col]),
                        index=X.index
                    )
                    X = pd.concat([X.drop(columns=[col]), one_hot_df], axis=1)

            elif method == "ordinal_encoder":
                if col in self.ordinal_encoders:
                    encoder = self.ordinal_encoders[col]
                    # Identify known and unknown categories
                    known_categories = encoder.categories_[0]
                    known_mask = X[col].isin(known_categories)
                    # Transform known categories
                    if known_mask.any():
                         X.loc[known_mask, col] = encoder.transform(X.loc[known_mask, [col]])[:, 0]
                    # Handle unknown categories
                    X.loc[~known_mask, col] = -1
                    X[col] = X[col].astype(int)

        return X


class NumericalNormalizer(BaseEstimator, TransformerMixin):
    """
    A transformer to normalize numerical columns in a dataset
    """
    def __init__(self, normalization_method_map: Dict[str, str]): 
        self.normalization_method_map = normalization_method_map or {}
        self.scalers = {}
        self.logger = get_logger(__name__, "PROD")

    def fit(self, X, y=None):
        for column, method in self.normalization_method_map.items():
            if method == "standard_scaler":
                scaler = StandardScaler()

            elif method == "minmax_scaler":
                scaler = MinMaxScaler()

            elif method == "robust_scaler":
                scaler = RobustScaler()

            elif method == "quantile_transformer":
                scaler = QuantileTransformer()

            elif method == "maxabs_scaler":
                scaler = MaxAbsScaler()

            elif method == "normalize_scaler":
                scaler = None

            else:
                self.logger.warning(f"Unknown method '{method}' for column '{column}'. Skipping.")
                continue

            if scaler is not None:
                scaler.fit(X[[column]])
                self.scalers[column] = scaler
            else:
                self.scalers[column] = None

        return self

    def transform(self, X):
        for column, scaler in self.scalers.items():
            if scaler is None:  # Directly use sklearn's normalize method
                X[column] = normalize(X[[column]], axis=0).flatten()  # Normalize to unit length
            else:
                X[column] = scaler.transform(X[[column]])

        return X


class FeatureEngineering:
    """
    A class for performing feature engineering on a dataset

    Parameters
    ----------
    data : pd.DataFrame
        The input data for the model training process
    
    target_col : str
        The target column name in the data

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
        Example usage: {'column_name1': 'mean', 'column_name2': 'mode'}

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
    """
    def __init__(
        self, 
        data: pd.DataFrame, 
        target_col: str, 
        drop_columns: Optional[List[str]] = None,
        categorical_imputation_method: str = "mode",
        numerical_imputation_method: str = "mean", 
        column_imputation_map: Optional[Dict[str, str]] = None,
        categorical_imputation_constant: str = "Unknown", 
        numerical_imputation_constant: float = 0.0,
        encoding_method: str = "onehot_encoder",
        onehot_limit: int = 25,
        encoding_method_map: Optional[Dict[str, str]] = None,
        ordinal_encode_map: Optional[Dict[str, List[str]]] = None,
        normalize: Optional[str] = None
    ):
        self.logger = get_logger(__name__, "PROD")

        # Initialize attributes
        self.data = data
        self.target_col = target_col
        self.drop_columns = drop_columns or []
        self.categorical_imputation_method = categorical_imputation_method
        self.numerical_imputation_method = numerical_imputation_method
        self.column_imputation_map = column_imputation_map or {}
        self.numerical_imputation_constant = numerical_imputation_constant
        self.categorical_imputation_constant = categorical_imputation_constant
        self.encoding_method = encoding_method
        self.onehot_limit = onehot_limit
        self.encoding_method_map = encoding_method_map or {}
        self.ordinal_encode_map = ordinal_encode_map or {}
        self.normalize = normalize
        self.y_class_mapping = None
        
    def setup(self, data: Optional[pd.DataFrame] = None):
        """
        Setup the feature engineering pipeline

        Parameters
        ----------
        data : pd.DataFrame, default=None
            The data to override the existing data attribute
        """
        if data is not None:
            self.data = data

        # Initialize encoder for target column
        self.target_encoder = LabelEncoder()
        # Separate features and target column
        self.feature_data = self.data.drop(columns=[self.target_col, *self.drop_columns], errors='ignore')
        self.numerical_columns = self.feature_data.select_dtypes(include=['number']).columns.tolist()
        self.categorical_columns = self.feature_data.columns.difference(self.numerical_columns).tolist()

        # Separate imputation mapping for numerical and categorical columns
        self.numerical_column_imputation_mapper = {
            col: self.numerical_imputation_method for col in self.numerical_columns
        }

        # For categorical columns, handle imputation separately
        self.categorical_column_imputation_mapper = {
            col: self.categorical_imputation_method for col in self.categorical_columns
        }

        # Combine both mappers to have a comprehensive imputation mapping
        self.column_imputation_mapper = {**self.numerical_column_imputation_mapper, 
                                         **self.categorical_column_imputation_mapper}
        
        # Update the mappers with any custom map provided
        if self.column_imputation_map:
            self.column_imputation_mapper.update(self.column_imputation_map)
        
        # Initialize encoding method mapper with default value and update with custom map
        self.encoding_method_mapper = {col: self.encoding_method for col in self.categorical_columns}
        if self.encoding_method_map:
            self.encoding_method_mapper.update(self.encoding_method_map)
        
        if self.ordinal_encode_map:
            for col in self.ordinal_encode_map.keys():
                if col in self.encoding_method_mapper:
                    self.encoding_method_mapper[col] = 'ordinal_encoder'
        
        # Initialize numerical normalization map
        if self.normalize:
            self.normalization_map = {
                col: self.normalize for col in self.numerical_columns
            }


        pipeline_steps = []

        # Add drop_columns step if drop_columns is not empty
        if self.drop_columns:
            pipeline_steps.append(("drop_columns", ColumnDropper(drop_columns=self.drop_columns)))

        # Add imputer step
        pipeline_steps.append(
            ("imputer", ColumnImputer(
                self.column_imputation_mapper, 
                self.numerical_imputation_constant, 
                self.categorical_imputation_constant
                )
            )
        )
        
        # Add normalization step if not None
        if self.normalize:
            pipeline_steps.append(("normalizer", NumericalNormalizer(self.normalization_map)))

        # Add encoding step
        pipeline_steps.append(("encoder", CategoricalEncoder(
            self.encoding_method_mapper, 
            self.ordinal_encode_map,
            onehot_limit=self.onehot_limit
        )))
        
        # Create the pipeline
        self.pipeline = Pipeline(pipeline_steps, memory=None)

    def check_column_anomalies(self, threshold: float = 0.5):
        """
        Identifies columns that are likely to be ID columns or have too many unique values

        Parameters
        ----------
        threshold : float 
            Threshold for the ratio (default is 0.5, e.g., 50%)
        """

        id_columns = self._id_finder()
        if id_columns:
            for column in id_columns:
                if column not in self.drop_columns:
                    self.logger.warning(f"Column '{column}' seems like an ID column. Consider dropping it via 'drop_columns' parameter if it is not a feature")

        columns_to_consider = self._anomaly_unique_values_finder(threshold=threshold)
        if columns_to_consider:
            for column, ratio in columns_to_consider.items():
                self.logger.warning(
                    f"Column '{column}' has too many unique values ({ratio:.2%}). "
                    "Recommended to either process or drop this column via 'drop_columns'"
                )

        # Find the columns that exceeds one_hot_limit
        columns_exceeding_limit = self._anomaly_onehot_limit_finder()
        # remove columns_to_consider from columns_exceeding_limit to avoid duplicate warnings
        columns_exceeding_limit = {k: v for k, v in columns_exceeding_limit.items() if k not in columns_to_consider}
        if columns_exceeding_limit:
            for column, count in columns_exceeding_limit.items():
                self.logger.warning(
                    f"Column '{column}' has {count} unique values. "
                    "Consider operations like increasing value of 'onehot_limit', "
                    "changing the encoding method or processing the column"
                )

    def _id_finder(self) -> list:
        """
        Identifies potential ID columns by checking if values in the first 100 rows 
        match their respective index values
        
        Returns
        -------
        list 
            List of column names that could be ID columns
        """
        potential_ids = []

        for column in self.data.columns:
            # Check if the first 100 rows match the index values
            if (self.data[column].iloc[:100] == self.data.index[:100]).all():
                potential_ids.append(column)
        
        return potential_ids

    def _anomaly_unique_values_finder(self, threshold: float = 0.5) -> dict:
        """
        Identifies categorical columns where the ratio of unique values to non-null rows
        exceeds the given threshold

        Parameters
        ----------
        threshold : float 
            Threshold for the ratio (default is 0.5, e.g., 50%)

        Returns
        -------
        dict
            Dictionary of column names and their unique value ratios
        """
        columns_above_threshold = {}

        for column in self.categorical_columns:
            # Calculate the ratio using non-null data
            non_null_count = self.data[column].notnull().sum()
            if non_null_count > 0:  # Avoid division by zero
                unique_ratio = self.data[column].nunique() / non_null_count
                if unique_ratio > threshold:
                    columns_above_threshold[column] = unique_ratio

        return columns_above_threshold
    
    def _anomaly_onehot_limit_finder(self) -> dict:
        """
        Identifies categorical columns where the number of unique values exceeds the one_hot_limit

        Returns
        -------
        dict
            Dictionary of column names and their unique value counts
        """
        columns_above_threshold = {}
        
        for column in self.categorical_columns:
            if self.data[column].nunique() > self.onehot_limit:
                columns_above_threshold[column] = self.data[column].nunique()

        return columns_above_threshold
    
    def fit_transform(self) -> pd.DataFrame:
        """
        Perform feature engineering on the training data

        Processes features and the target column by:
        - Dropping specified columns from the data
        - Imputing missing values for numerical and categorical columns
        - Encoding categorical features
        - Encoding the target column if it is categorical
        - Normalizing numerical columns if specified

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the processed features and target column
        """
        # Process features
        processed_features = self.pipeline.fit_transform(self.feature_data)
        
        # Process if target column is categorical
        target_data = self.data[self.target_col]
        if target_data.dtype in ['object', 'category']:
            target_data = self.target_encoder.fit_transform(target_data)
            self.y_class_mapping = { # for example: {0: 'male', 1: 'female'}
                i: label for i, label in enumerate(self.target_encoder.classes_)
            }
        processed_features[self.target_col] = target_data

        return processed_features.drop(self.target_col, axis=1), processed_features[self.target_col]

    def transform(self, test_data: pd.DataFrame, y_included: bool = False) -> pd.DataFrame:
        """
        Perform feature engineering on test data using the fitted pipeline

        Processes features by:
        - Imputing missing values for numerical and categorical columns
        - Encoding categorical features
        - Normalizing numerical columns if specified

        Parameters
        ----------
        test_data : pd.DataFrame
            The test dataset to process

        y_included : bool, default=False
            Whether the target column is included in the test data so It also transforms the target column

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the processed test features
        """
        if y_included:
            test_features = test_data
        else:
            test_features = test_data.drop(columns=[self.target_col], errors='ignore')

        processed_test_features = self.pipeline.transform(test_features)
        
        # Add target column if it exists in test data
        if self.target_col in test_data.columns:
            target_data = test_data[self.target_col]
            if target_data.dtype in ['object', 'category']:
                target_data = self.target_encoder.transform(target_data)
            processed_test_features[self.target_col] = target_data

        if not y_included:
            return processed_test_features
        else:
            return processed_test_features.drop(self.target_col, axis=1), processed_test_features[self.target_col]