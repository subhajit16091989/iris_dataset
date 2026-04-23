from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pandas as pd
from src.features.preprocessor import build_preprocessor
from sklearn.decomposition import PCA

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, model_params: dict, scale_nummeric: True,pca_enable: bool, pca_n_components: float) -> Pipeline:
    
    preprocessor = build_preprocessor(X_train, scale_numeric=scale_nummeric)
    model = LogisticRegression(**model_params)
    steps = [("preprocessor", preprocessor)]
    if pca_enable:
        steps.append(("pca", PCA(n_components=pca_n_components)))
    steps.append(("model", model))
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)
    return pipeline