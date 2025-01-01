import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from typing import List, Optional, Dict, Any


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    A transformer to drop specified columns from a dataset.
    """
    def __init__(self, drop_columns: Optional[List[str]] = None):
        self.drop_columns = drop_columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Drops specified columns from the input DataFrame.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame with the specified columns dropped.
        """
        return X.drop(columns=self.drop_columns, axis=1, errors='ignore')
    

class ColumnImputer(BaseEstimator, TransformerMixin):
    """
    A transformer to impute missing values in a dataset.
    """
    def __init__(
        self, 
        column_imputation_mapper, 
        numerical_imputation_constant=0.0, 
        categorical_imputation_constant="Unknown"
        ):

        self.column_imputation_mapper = column_imputation_mapper
        self.numerical_imputation_constant = numerical_imputation_constant
        self.categorical_imputation_constant = categorical_imputation_constant

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> pd.DataFrame:
        """
        Impute missing values in the input DataFrame based on the specified methods.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to impute missing values.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame with missing values imputed based on the specified methods
        """
        X = X.copy()
        for column, method in self.column_imputation_mapper.items():
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
    A transformer to encode categorical columns in a dataset.
    """
    def __init__(
        self, 
        encoding_method_mapper: Optional[Dict[str, Any]] = None, 
        ordinal_map: Optional[Dict[str, Any]] = None,
        onehot_limit=25
        ):

        self.encoding_method_mapper = encoding_method_mapper or {}
        self.ordinal_map = ordinal_map or {}
        self.onehot_limit = onehot_limit
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.ordinal_encoders = {}

    def fit(self, X, y=None):
        """
        Fit the encoder on the input DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to fit the encoder on.
        """
        for col, method in self.encoding_method_mapper.items():  # Only iterate over mapped columns
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
        """
        Transform the input DataFrame by encoding categorical columns.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to transform.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame with the categorical columns encoded.
        """
        X = X.copy()
        for col, method in self.encoding_method_mapper.items():  # Only iterate over mapped columns
            if method == "label_encoder":
                if col in self.label_encoders:
                    encoder = self.label_encoders[col]
                    X[col] = X[col].apply(
                        lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                    )
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
                    categories = encoder.categories_[0]
                    X[col] = X[col].apply(
                        lambda x: encoder.transform(pd.DataFrame([[x]], columns=[col]))[0][0] if x in categories else -1
                    )
        return X


class NumericalNormalizer(BaseEstimator, TransformerMixin):
    """
    A transformer to normalize numerical columns in a dataset.
    """
    def __init__(self, normalization_method_map: Optional[Dict[str, str]] = None):
        self.normalization_method_map = normalization_method_map or {}
        self.scalers = {}

    def fit(self, X, y=None):
        """
        Fit the normalizer on the input DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to fit the normalizer
        """
        for column, method in self.normalization_method_map.items():
            if method == "standard_scaler":
                scaler = StandardScaler()
            elif method == "normalize_scaler":
                scaler = MinMaxScaler()
            elif method == "robust_scaler":
                scaler = RobustScaler()
            elif method == "quantile_transformer":
                scaler = QuantileTransformer()
            else:
                print(f"Warning: Unknown method '{method}' for column '{column}'. Skipping.") # This shouldn't be needed after validator but just in case
                continue
            scaler.fit(X[[column]])
            self.scalers[column] = scaler
        return self

    def transform(self, X):
        """
        Transform the input DataFrame by normalizing numerical columns.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the numerical columns normalized.
        """
        X = X.copy()
        for column, scaler in self.scalers.items():
            X[column] = scaler.transform(X[[column]])
        return X


class FeatureEngineering:
    """
    A class for performing feature engineering on a dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The input data for the model training process
    
    target_col : str
        The target column name in the data

    drop_columns : list, default=None
        Columns that will be dropped from the data.
    
    categorical_imputation_method : str, default='mode'
        Imputation method for categorical columns. Options:
        * 'mode': Replace missing values with the most frequent value.
        * 'constant': Replace missing values with a constant value.
        * 'drop': Drop rows with missing values.

    numerical_imputation_method : str, default='mean'
        Imputation method for numerical columns. Options:
        * 'mean': Replace missing values with the column mean.
        * 'median': Replace missing values with the column median.
        * 'mode': Replace missing values with the column mode.
        * 'constant': Replace missing values with a constant value.
        * 'drop': Drop rows with missing values.

    column_imputation_map : dict, default=None
        Custom mapping of columns to specific imputation methods.
        Example usage: {'column_name': 'mean', 'column_name2': 'mode'}

    numerical_imputation_constant : float, default=0.0
        The constant value for imputing numerical columns when 'constant' is selected.

    categorical_imputation_constant : str, default='Unknown'
        The constant value for imputing categorical columns when 'constant' is selected.

    encoding_method : str, default='label_encoder'
        Encoding method for categorical columns. Options:
        * 'label_encoder': Use label encoding.
        * 'onehot_encoder': Use one-hot encoding.
        * 'ordinal_encoder': Use ordinal encoding.
        
    onehot_limit : int, default=25
        Maximum number of categories to use for one-hot encoding.

    encoding_method_map : dict, default=None
        Custom mapping of columns to encoding methods.
        Example usage: {'column_name': 'onehot_encoder', 'column_name2': 'label_encoder'}
    
    ordinal_encode_map : dict, default=None
        Custom mapping of columns to category order for ordinal encoding.
        Example usage: {'column_name': ['low', 'medium', 'high']}
    
    normalize_numerical : str, default=None
        Standardize the data using StandardScaler. Options:
        * 'standard_scaler': Standardize the data.
        * 'normalize_scaler': Normalize the data.
        * 'robust_scaler': Scale the data using RobustScaler.
        * 'quantile_transformer': Transform the data using QuantileTransformer.
    """
    def __init__(
        self, 
        data: pd.DataFrame, 
        target_col: str, 
        drop_columns: Optional[List[str]] = None,
        categorical_imputation_method: str = "mode",
        numerical_imputation_method: str = "mean", 
        column_imputation_map: Optional[Dict[str, str]] = None,
        numerical_imputation_constant: float = 0.0,
        categorical_imputation_constant: str = "Unknown", 
        encoding_method: str = "label_encoder",
        onehot_limit: int = 25,
        encoding_method_map: Optional[Dict[str, str]] = None,
        ordinal_encode_map: Optional[Dict[str, List[str]]] = None,
        normalize_numerical: Optional[str] = None
        ):

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
        self.normalize_numerical = normalize_numerical
        # Initialize encoder for target column
        self.target_encoder = LabelEncoder()
        # Separate features and target column
        self.feature_data = self.data.drop(columns=[self.target_col, *self.drop_columns], errors='ignore')
        self.numerical_columns = self.feature_data.select_dtypes(include=['number']).columns.tolist()
        self.categorical_columns = self.feature_data.columns.difference(self.numerical_columns).tolist()
        self.feature_data[self.categorical_columns] = self.feature_data[self.categorical_columns].astype(str)

        # Separate imputation mapping for numerical and categorical columns
        self.numerical_column_imputation_mapper = {
            col: self.numerical_imputation_method for col in self.numerical_columns
        }

        # For categorical columns, handle imputation separately
        self.categorical_column_imputation_mapper = {
            col: self.categorical_imputation_method for col in self.categorical_columns
        }

        # Update the mappers with any custom map provided
        if self.column_imputation_map:
            for col, method in self.column_imputation_map.items():
                if col in self.numerical_columns:
                    self.numerical_column_imputation_mapper[col] = method
                elif col in self.categorical_columns:
                    self.categorical_column_imputation_mapper[col] = method

        # Combine both mappers to have a comprehensive imputation mapping
        self.column_imputation_mapper = {**self.numerical_column_imputation_mapper, 
                                         **self.categorical_column_imputation_mapper}
        
        # Initialize encoding method mapper with default value and update with custom map
        self.encoding_method_mapper = {col: self.encoding_method for col in self.categorical_columns}
        if self.encoding_method_map:
            self.encoding_method_mapper.update(encoding_method_map)
        
        # Initialize numerical normalization map
        self.numerical_normalization_map = normalize_numerical
        if self.normalize_numerical:
            self.numerical_normalization_map = {
                col: normalize_numerical for col in self.numerical_columns if normalize_numerical is not None
            }

        # Create the feature engineering pipeline
        pipeline_steps = [
            ("drop_columns", ColumnDropper(drop_columns=self.drop_columns)),
            ("imputer", ColumnImputer(
                self.column_imputation_mapper, 
                self.numerical_imputation_constant, 
                self.categorical_imputation_constant
            )),
        ]
        
        # Add normalization step if not None
        if self.normalize_numerical:
            pipeline_steps.append(("normalizer", NumericalNormalizer(self.numerical_normalization_map)))

        # Add encoding step
        pipeline_steps.append(("encoder", CategoricalEncoder(
            self.encoding_method_mapper, 
            self.ordinal_encode_map,
            onehot_limit=self.onehot_limit
        )))
        
        # Create the pipeline
        self.pipeline = Pipeline(pipeline_steps)

    def start_feature_engineering(self) -> pd.DataFrame:
        """
        Perform feature engineering on the training data.

        Processes features and the target column by:
        - Dropping specified columns from the data.
        - Imputing missing values for numerical and categorical columns.
        - Encoding categorical features.
        - Encoding the target column if it is categorical.
        - Normalizing numerical columns if specified.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the processed features and target column.
        """
        # Process features
        processed_features = self.pipeline.fit_transform(self.feature_data)
        
        # Process if target column is categorical
        target_data = self.data[self.target_col]
        if target_data.dtype == 'object':
            target_data = self.target_encoder.fit_transform(target_data)
        processed_features[self.target_col] = target_data
        return processed_features

    def start_feature_engineering_test_data(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on test data using the fitted pipeline.

        Processes features by:
        - Imputing missing values for numerical and categorical columns.
        - Encoding categorical features.
        - Normalizing numerical columns if specified.

        Parameters
        ----------
        test_data : pd.DataFrame
            The test dataset to process.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the processed test features.
        """
        test_features = test_data.drop(columns=[self.target_col], errors='ignore')
        processed_test_features = self.pipeline.transform(test_features)
        
        # Add target column if it exists in test data
        if self.target_col in test_data.columns:
            target_data = test_data[self.target_col]
            if target_data.dtype == 'object':
                target_data = self.target_encoder.transform(target_data)
            processed_test_features[self.target_col] = target_data
            
        return processed_test_features