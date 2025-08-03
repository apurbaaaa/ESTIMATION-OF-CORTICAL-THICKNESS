"""Model definition for cortical thickness estimation."""

from sklearn.ensemble import RandomForestRegressor


def get_model(random_state: int = 42) -> RandomForestRegressor:
    """Return a new ``RandomForestRegressor`` instance.

    Parameters
    ----------
    random_state: int, optional
        Random seed for reproducibility.
    """

    return RandomForestRegressor(n_estimators=100, random_state=random_state)


__all__ = ["get_model"]
