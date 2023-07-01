from typing import Optional
import logging
import numpy as np
import pandas as pd
import sys


APPLICATION_NAME = "MyLineReg"
logger = logging.getLogger(APPLICATION_NAME)


class MyLineReg():

    REPR_STR = (
        '{class_name} class: n_iter={n_iter}, learning_rate={learning_rate}'
    )
    LOG_STR = '{start} | loss: {loss}'

    def __init__(self, n_iter=100, learning_rate=0.1) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = np.array([])
        self._setup_logging()

    def __repr__(self) -> str:
        return self.REPR_STR.format(
            class_name=self.__class__.__name__,
            n_iter=self.n_iter,
            learning_rate=self.learning_rate
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
        for i in range(self.n_iter):
            y_hat = features @ self.weights
            loss = y_hat - y
            mse_loss = (loss.to_numpy() ** 2).sum() / loss.size
            if verbose and not i % verbose:
                self._print_train_log(i, mse_loss)
            grad = loss @ features * 2 / features.shape[0]
            self.weights = self.weights - self.learning_rate * grad.to_numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        features = self._add_bias(X)
        return (features @ self.weights).to_numpy()

    def get_coef(self) -> Optional[np.ndarray]:
        if self.weights.shape[0] > 1:
            return self.weights[1:]
        else:
            return None

    def _add_bias(self, features: pd.DataFrame) -> pd.DataFrame:
        features_with_bias = features.copy()
        features_with_bias.insert(loc=0, column='bias', value=1)
        return features_with_bias

    def _initialize_weights(self, features: pd.DataFrame) -> None:
        self.weights = np.ones(features.shape[1])

    def _print_train_log(self, i: int, loss: float) -> None:
        logger.info(self.LOG_STR.format(start=i if i else 'start', loss=loss))
