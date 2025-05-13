from flwr.client import NumPyClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import numpy as np

class SklearnClient(NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.model = LinearRegression()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, config):
        try:
            check_is_fitted(self.model)
            return [self.model.coef_, self.model.intercept_]

        except NotFittedError:
            n_features = self.X_train.shape[1]
            return [np.zeros((n_features,)), 0.0]


    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        return self.get_parameters(config={}), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        preds = self.model.predict(self.X_test)
        loss = mean_squared_error(self.y_test, preds)
        r2 = r2_score(self.y_test, preds)
        return float(loss), len(self.X_test), {"r2": float(r2)}
