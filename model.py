"""Model factory for federated cortical thickness estimation.

This module exposes :func:`get_model` which instantiates one of the
regressors used throughout the original notebook pipeline.  The factory
supports Random Forest, XGBoost, CatBoost, and k-NN regressors so the
federated simulation can be run with each of them.
"""

from __future__ import annotations

from typing import Literal

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

try:  # Import optional third‑party regressors
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - xgboost may be absent at import time
    XGBRegressor = None  # type: ignore

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - catboost may be absent at import time
    CatBoostRegressor = None  # type: ignore


ModelName = Literal["random_forest", "xgboost", "catboost", "knn"]


def get_model(name: ModelName = "random_forest", random_state: int | None = 42):
    """Return a regressor instance specified by ``name``.

    Parameters
    ----------
    name:
        Identifier of the model to create.  One of ``"random_forest"``,
        ``"xgboost"``, ``"catboost"`` or ``"knn"``.
    random_state:
        Random seed for reproducibility where applicable.
    """

    if name == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=random_state)
    if name == "xgboost":
        if XGBRegressor is None:  # pragma: no cover
            raise ImportError("xgboost is not installed")
        return XGBRegressor(objective="reg:squarederror", random_state=random_state)
    if name == "catboost":
        if CatBoostRegressor is None:  # pragma: no cover
            raise ImportError("catboost is not installed")
        return CatBoostRegressor(verbose=0, random_state=random_state)
    if name == "knn":
        return KNeighborsRegressor(n_neighbors=10)

    raise ValueError(f"Unknown model name: {name}")


__all__ = ["get_model", "ModelName"]
