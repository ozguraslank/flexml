"""
FlexML: Easy-to-use and flexible AutoML library for Python
"""

from flexml.helpers.tools import check_numpy_dtype_error
check_numpy_dtype_error() # Check cronic Colab version issue

from .regression import Regression
from .classification import Classification

__version__ = "1.1.0rc0"

__all__ = ["Regression", "Classification"]
