from typing import Union, Callable
import inspect
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer


class CustomScore:
    def __init__(
        self,
        name: str,
        score_func: Callable,
        needs_proba: bool,
        direction: str
    ):
        self.name = name
        self.score_func = score_func
        self.needs_proba = needs_proba
        self.direction = direction

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"name must be a non-empty string, got '{self.name}'")

        if direction not in ['maximize', 'minimize']:
            raise ValueError(f"direction must be either 'maximize' or 'minimize', got '{direction}'")

        if needs_proba is None or not isinstance(needs_proba, bool):
            raise ValueError(f"needs_proba must be a boolean, got '{needs_proba}'")

        try:
            sig = inspect.signature(score_func)
            params = list(sig.parameters.keys())
            
            # Check if function has exactly 2 parameters
            if len(params) != 2:
                raise ValueError(
                    f"Custom evaluation function must have exactly 2 parameters (y_true, y_pred), "
                    f"but got {len(params)} parameters: {params}"
                )
        except Exception as e:
            raise ValueError(f"Error validating custom evaluation function: {str(e)}")

        self.scorer = make_scorer(
            self.score_func,
            needs_proba=self.needs_proba,
            greater_is_better=self.direction == 'maximize'
        )

    def __call__(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
        return self.score_func(y_true, y_pred)

    def __repr__(self):
        return f"CustomScore(name={self.name}, score_func={self.score_func.__name__}, needs_proba={self.needs_proba}, direction={self.direction})"

    def get_scorer(self):
        return self.scorer