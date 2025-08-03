from flwr.client import NumPyClient
from sklearn.metrics import mean_squared_error
import numpy as np

class SklearnClient(NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, target):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train[target]
        self.X_test = X_test
        self.y_test = y_test[target]
        self.target = target

    def get_parameters(self, config):
        return self.model.get_params(deep=True)

    def fit(self, parameters, config):
        self.model.set_params(**parameters)
        self.model.fit(self.X_train, self.y_train)
        return self.model.get_params(deep=True), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_params(**parameters)
        predictions = self.model.predict(self.X_test)
        loss = mean_squared_error(self.y_test, predictions)
        return float(loss), len(self.X_test), {"mse": float(loss)}
