from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pandas as pd
from src.features.preprocessor import build_preprocessor

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, model_params: dict, scale_nummeric: True) -> Pipeline:
    
    preprocessor = build_preprocessor(X_train, scale_numeric=scale_nummeric)
    model = LogisticRegression(**model_params)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline