from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import logging

def build_preprocessor(X: pd.DataFrmame, scale_numeric: bool = True) -> ColumnTransformer:
    if X.empty:
        raise ValueError("Input DataFrame is empty.")
    logger = logging.getLogger(__name__)
    logger.info("Building preprocessing pipeline")

    # Identify feature groups
    numerirc_features = X.select_dtypes(include=['number', 'bool']).columns.tolist()
    categogorical_features = [c for c in X.columns if c not in numerirc_features]
    logger.info(f"Numeric features identified: {numerirc_features}")
    logger.info(f"Categorical features identified: {categogorical_features}")

    # Numeric pipeline
    numeric_steps = [("imputer", SimpleImputer(strategy='median'))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    logger.info("Numeric pipeline built.")

    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("encoder", OneHotEncoder(handle_unknown='ignore'))
    ])
    logger.info("Categorical pipeline built.")

    # Combine pipelines
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numerirc_features),
        ("categorical", categorical_pipeline, categogorical_features)
    ])
    logger.info("Preprocessor built successfully.")

    return preprocessor