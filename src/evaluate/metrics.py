"""Evaluation metric utilities for binary classification tasks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    unique_classes = np.unique(y_true)
    is_multiclass = len(unique_classes) > 2

    if is_multiclass:
        avg = "macro"   # or "weighted"
    else:
        avg = "binary"

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
    }
    if y_prob is not None:
        unique = np.unique(y_true)
        # ROC-AUC is only defined for binary targets in this helper.
        if len(unique) == 2:
              # Binary case
            if y_prob.ndim == 2:
                y_prob_bin = y_prob[:, 1]  # extract positive class
            else:
                y_prob_bin = y_prob
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob_bin))
        else:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    return out