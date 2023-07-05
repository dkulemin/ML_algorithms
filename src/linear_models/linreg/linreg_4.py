from typing import Optional, Callable
import logging
import numpy as np
import pandas as pd
import sys


APPLICATION_NAME = "MyLineReg"
logger = logging.getLogger(APPLICATION_NAME)


class MyLineReg():
    @staticmethod
    def _mean_squared_error(y_true, y_pred) -> float:
        loss = y_true - y_pred
        return (loss ** 2).sum() / y_pred.size

    @staticmethod
    def _root_mean_squared_error(y_true, y_pred) -> float:
        return MyLineReg._mean_squared_error(y_true, y_pred) ** 0.5

    @staticmethod
    def _mean_absolute_error(y_true, y_pred) -> float:
        loss = y_true - y_pred
        return np.abs(loss).sum() / y_pred.size

    @staticmethod
    def _mean_absolute_percentage_error(y_true, y_pred) -> float:
        loss = (y_true - y_pred) / y_true
        return np.abs(loss).sum() / y_pred.size * 100

    @staticmethod
    def _r2_score(y_true, y_pred) -> float:
        loss = y_true - y_pred
        y_normed = y_true - y_true.mean()
        return 1 - (loss ** 2).sum() / (y_normed ** 2).sum()

    METRIC_CALLABLE = {
            'mae': _mean_absolute_error,
            'mse': _mean_squared_error,
            'rmse': _root_mean_squared_error,
            'mape': _mean_absolute_percentage_error,
            'r2': _r2_score,
        }

    REPR_STR = (
        '{class_name} class: n_iter={n_iter}, '
        'learning_rate={learning_rate}, metric={metric}'
    )
    LOG_STR = '{start} | loss: {loss}'
    LOG_METRIC_STR = '{start} | loss: {loss} | {metric_name}: {metric}'

    def __init__(self, n_iter=100, learning_rate=0.1, metric=None) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.weights = np.array([])
        self.metric_loss = 0.
        self._setup_logging()

    def __repr__(self) -> str:
        return self.REPR_STR.format(
            class_name=self.__class__.__name__,
            n_iter=self.n_iter,
            learning_rate=self.learning_rate,
            metric=self.metric
        )

    def _setup_logging(self) -> None:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter(fmt="%(message)s"))

        logger = logging.getLogger(APPLICATION_NAME)
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.addHandler(sh)

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False) -> None:
        features = self._add_bias(X)
        self._initialize_weights(features)
        metric_clbl = self.METRIC_CALLABLE.get(self.metric, None)
        for i in range(self.n_iter):
            self._calculate_weights(features, y)
            mse_loss = self._calculate_metrics(features, y, metric_clbl)
            if verbose and not i % verbose:
                if metric_clbl:
                    self._print_train_log(i, mse_loss, self.metric_loss)
                else:
                    self._print_train_log(i, mse_loss)

    def _calculate_weights(self, features: pd.DataFrame, y: pd.Series) -> None:
        y_pred = features @ self.weights
        loss = y_pred - y
        grad = loss @ features * 2 / features.shape[0]
        self.weights = self.weights - self.learning_rate * grad.to_numpy()

    def _calculate_metrics(
        self,
        features: pd.DataFrame,
        y: pd.Series,
        metric_clbl: Optional[Callable] = None
    ) -> float:
        y_pred = features @ self.weights
        mse_loss = self._mean_squared_error(y, y_pred)
        self.metric_loss = metric_clbl(y, y_pred) if metric_clbl else mse_loss
        return mse_loss

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        features = self._add_bias(X)
        return (features @ self.weights).to_numpy()

    def get_coef(self) -> Optional[np.ndarray]:
        if self.weights.shape[0] > 1:
            return self.weights[1:]
        else:
            return None

    def get_best_score(self) -> float:
        return self.metric_loss

    def _add_bias(self, features: pd.DataFrame) -> pd.DataFrame:
        features_with_bias = features.copy()
        features_with_bias.insert(loc=0, column='bias', value=1)
        return features_with_bias

    def _initialize_weights(self, features: pd.DataFrame) -> None:
        self.weights = np.ones(features.shape[1])

    def _print_train_log(
        self, i: int, loss: float, metric: Optional[float] = None
    ) -> None:
        if self.metric:
            logger.info(
                self.LOG_METRIC_STR.format(
                    start=i if i else 'start',
                    loss=loss,
                    metric_name=self.metric,
                    metric=metric
                )
            )
        else:
            logger.info(
                self.LOG_STR.format(
                    start=i if i else 'start',
                    loss=loss
                )
            )
