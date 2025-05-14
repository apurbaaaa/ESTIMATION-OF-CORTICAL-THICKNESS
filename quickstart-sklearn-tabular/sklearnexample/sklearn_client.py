from flwr.client import NumPyClient
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import numpy as np

class SklearnMultiOutputClient(NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test, target_names):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.target_names = target_names

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
        # Fit per target column
        for col in self.target_names:
            self.model.fit(self.X_train, self.y_train[col])
        return self.get_parameters(config={}), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = {}
        total_loss = 0

        for col in self.target_names:
            self.model.fit(self.X_train, self.y_train[col])  # Refit for each target
            preds = self.model.predict(self.X_test)
            loss = mean_squared_error(self.y_test[col], preds)
            r2 = r2_score(self.y_test[col], preds)
            acc = self.model.score(self.X_test, self.y_test[col])  # same as R²
            metrics[f"{col}_mse"] = float(loss)
            metrics[f"{col}_r2"] = float(r2)
            metrics[f"{col}_accuracy"] = float(acc)
            total_loss += loss

        avg_loss = total_loss / len(self.target_names)
        return avg_loss, len(self.X_test), metrics
