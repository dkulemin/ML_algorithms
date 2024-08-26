from mock import patch  # type: ignore
from mock.mock import PropertyMock  # type: ignore
import logging
import math
import pytest

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
import numpy as np
import pandas as pd

from src.linear_models.linreg.linreg_5 import MyLineReg, APPLICATION_NAME


logger = logging.getLogger(APPLICATION_NAME)


@pytest.fixture(scope="module")
def make_regression_data():
    X, y = make_regression(
        n_samples=1000,
        n_features=14,
        n_informative=1,
        noise=15,
        random_state=42
    )
    X = pd.DataFrame(X)
    y = pd.Series(y)
    return X, y


@pytest.fixture(scope="module")
def regressor(make_regression_data):
    X, y = make_regression_data
    regressor = LinearRegression()
    regressor.fit(X, y)
    return regressor


def test_can_import_class():
    from src.linear_models.linreg.linreg_5 import MyLineReg  # noqa: F401


@pytest.mark.parametrize(
    'n_iter, etalon_n_iter',
    [
        pytest.param(None, 100, id='default'),
        pytest.param(10, 10, id='set_param'),
    ]
)
def test_class_initialization_n_iter(n_iter, etalon_n_iter):
    linreg = MyLineReg(n_iter) if n_iter else MyLineReg()
    assert linreg.n_iter == etalon_n_iter, (
        f'n_iter should be {etalon_n_iter}, but got {linreg.n_iter}'
    )


@pytest.mark.parametrize(
    'learning_rate, etalon_lr',
    [
        pytest.param(None, 0.1, id='default'),
        pytest.param(0.001, 0.001, id='set_param'),
    ]
)
def test_class_initialization_learning_rate(learning_rate, etalon_lr):
    linreg = (
        MyLineReg(learning_rate=learning_rate)
        if learning_rate
        else MyLineReg()
    )
    assert linreg.learning_rate == etalon_lr, (
        f'n_iter should be {etalon_lr}, but got {linreg.learning_rate}'
    )


def test_class_initialization_metric():
    for metric in [None, 'mae', 'mse', 'rmse', 'mape', 'r2']:
        linreg = (
            MyLineReg(metric=metric)
            if metric
            else MyLineReg()
        )
        assert linreg.metric == metric, (
            f'n_iter should be {metric}, but got {linreg.metric}'
        )


def test_class_initialization_reg():
    for reg in [None, 'l1', 'l2', 'elasticnet']:
        linreg = MyLineReg(reg=reg) if reg else MyLineReg()
        assert linreg.reg == reg, (
            f'n_iter should be {reg}, but got {linreg.reg}'
        )


def test_class_initialization_l1_coef():
    for l1_coef in np.linspace(0., 1., num=11, endpoint=True):
        linreg = MyLineReg(l1_coef=l1_coef) if l1_coef else MyLineReg()
        assert linreg.l1_coef == l1_coef, (
            f'n_iter should be {l1_coef}, but got {linreg.l1_coef}'
        )


def test_class_initialization_l2_coef():
    for l2_coef in np.linspace(0., 1., num=11, endpoint=True):
        linreg = MyLineReg(l2_coef=l2_coef) if l2_coef else MyLineReg()
        assert linreg.l2_coef == l2_coef, (
            f'n_iter should be {l2_coef}, but got {linreg.l2_coef}'
        )


@pytest.mark.parametrize(
    (
        'n_iter, learning_rate, metric, '
        'default_n_iter, default_lr, default_metric'
    ),
    [
        pytest.param(None, None, None, 100, 0.1, None, id='all_defaults'),
        pytest.param(10, None, None, 100, 0.1, None, id='set_n_iter'),
        pytest.param(
            None, 0.001, None, 100, 0.1, None, id='set_learning_rate'
        ),
        pytest.param(
            None, None, 'mae', 100, 0.1, 'mae', id='set_mae_metric'
        ),
        pytest.param(
            None, None, 'mse', 100, 0.1, 'mse', id='set_mse_metric'
        ),
        pytest.param(
            None, None, 'rmse', 100, 0.1, 'rmse', id='set_rmse_metric'
        ),
        pytest.param(
            None, None, 'mape', 100, 0.1, 'mape', id='set_mape_metric'
        ),
        pytest.param(
            None, None, 'r2', 100, 0.1, 'r2', id='set_r2_metric'
        ),
        pytest.param(10, 0.001, 'mae', 100, 0.1, 'mae', id='set_all_params'),
    ]
)
@patch.object(MyLineReg, 'REPR_STR', new_callable=PropertyMock)
def test_class_representation(
    mock_repr,
    n_iter,
    learning_rate,
    metric,
    default_n_iter,
    default_lr,
    default_metric
):
    etalon_n_iter = n_iter if n_iter else default_n_iter
    etalon_learning_rate = learning_rate if learning_rate else default_lr
    etalon_metric = metric if metric else default_metric

    representation_string = (
        '{class_name}: n_iter={n_iter}, lr={learning_rate}, metric={metric}'
    )
    mock_repr.return_value = representation_string
    linreg = MyLineReg(etalon_n_iter, etalon_learning_rate, etalon_metric)
    result = repr(linreg)
    mock_repr.assert_called()
    etalon_representation = representation_string.format(
        class_name=linreg.__class__.__name__,
        n_iter=etalon_n_iter,
        learning_rate=etalon_learning_rate,
        metric=metric,
    )

    assert result == etalon_representation, (
        f'class representation should be {etalon_representation} '
        f'but got {result}'
    )


def test_get_coef_no_weights():
    linreg = MyLineReg()
    result = linreg.get_coef()
    assert result is None, 'returned weights should be `None`'


def test_get_coef_wrong_weights_number():
    linreg = MyLineReg()
    for _ in range(10):
        linreg.weights = np.random.rand(1)
        result = linreg.get_coef()
        assert result is None, 'returned weights should be `None`'


def test_get_coef():
    linreg = MyLineReg()
    for _ in range(10):
        weights = np.random.rand(101)
        linreg.weights = weights
        result = linreg.get_coef()
        etalon_weights = weights[1:]
        assert np.array_equal(result, etalon_weights), (
            f'returned weights should be {etalon_weights}, but got {result}'
        )


def test_add_bias(make_regression_data):
    features, _ = make_regression_data
    etalon_features = features.copy()
    etalon_features.insert(loc=0, column='bias', value=1)

    linreg = MyLineReg()
    result_features = linreg._add_bias(features)

    assert etalon_features.shape[1] == result_features.shape[1], (
        'number of features should be features + 1'
    )

    first_col_values = result_features.iloc[:, 0]

    assert first_col_values.nunique() == 1, (
        'there are should be only one value in first column'
    )

    assert (first_col_values.to_numpy() == 1).all(), (
        'all values in first column are 1s'
    )


def test_initialize_weights(make_regression_data):
    features, _ = make_regression_data

    linreg = MyLineReg()
    linreg._initialize_weights(features)
    result_weights = linreg.weights

    assert isinstance(result_weights, np.ndarray), (
        'weights should be numpy array type'
    )
    assert features.shape[1] == result_weights.shape[0], (
        'weights size should be features + 1'
    )
    assert (result_weights == 1).all(), 'all weights should be 1'


@patch.object(MyLineReg, 'LOG_STR', new_callable=PropertyMock)
def test_print_train_log(mock_log, caplog):
    log_string = '{start} iter | loss: {loss}'
    mock_log.return_value = log_string
    iters = np.arange(0, 10)
    losses = np.linspace(3, 0, num=10, endpoint=False)

    linreg = MyLineReg()
    caplog.set_level(logging.INFO)
    for i, loss in zip(iters, losses):
        linreg._print_train_log(i, loss)
        mock_log.assert_called()
        etalon_log_string = log_string.format(
            start=i if i else 'start',
            loss=loss
        )
        assert any(
            etalon_log_string in message for message in caplog.messages
        ), f'log string `{etalon_log_string}` should be in {caplog.messages},'
    assert mock_log.call_count == iters.size, (
        f'should be excectly {iters.size}, but got {mock_log.call_count}'
    )


@pytest.mark.parametrize(
    'metric_name',
    [
        pytest.param('mae', id='mae_metric'),
        pytest.param('mse', id='mse_metric'),
        pytest.param('rmse', id='rmse_metric'),
        pytest.param('mape', id='mape_metric'),
        pytest.param('r2', id='r2_metric'),
    ]
)
@patch.object(MyLineReg, 'LOG_METRIC_STR', new_callable=PropertyMock)
def test_print_train_log_with_metric(mock_log_metric, metric_name, caplog):
    mock_log_metric.reset_mock()
    log_metric_string = '{start} iter | loss: {loss} | {metric_name}: {metric}'
    mock_log_metric.return_value = log_metric_string
    iters = np.arange(0, 10)
    losses = np.linspace(3, 0, num=10, endpoint=False)
    metrics = np.linspace(5, 3, num=10, endpoint=False)

    linreg = MyLineReg(metric=metric_name)
    caplog.set_level(logging.INFO)
    for i, loss, metric in zip(iters, losses, metrics):
        linreg._print_train_log(i, loss, metric)
        mock_log_metric.assert_called()
        etalon_log_string = log_metric_string.format(
            start=i if i else 'start',
            loss=loss,
            metric_name=linreg.metric,
            metric=metric
        )
        assert any(
            etalon_log_string in message for message in caplog.messages
        ), f'log string `{etalon_log_string}` should be in {caplog.messages},'
    etalon_call_count = mock_log_metric.call_count
    assert etalon_call_count == iters.size, (
        f'should be excectly {iters.size}, but got {etalon_call_count}'
    )


def test_fit(make_regression_data, regressor):
    X, y = make_regression_data
    etalon_weights = np.array([regressor.intercept_] + list(regressor.coef_))

    linreg = MyLineReg()
    linreg.fit(X, y)
    result_weghts = linreg.weights

    assert np.allclose(
        result_weghts, etalon_weights, rtol=1e-05, atol=1e-08
    ), 'weights should be close enough to weights from sklearn model'


@pytest.mark.parametrize(
    'verbose',
    [
        pytest.param(False, id='verbose_off'),
        pytest.param(1, id='verbose_1'),
        pytest.param(2, id='verbose_2'),
        pytest.param(3, id='verbose_3'),
        pytest.param(4, id='verbose_4'),
        pytest.param(5, id='verbose_5'),
        pytest.param(6, id='verbose_6'),
        pytest.param(7, id='verbose_7'),
        pytest.param(8, id='verbose_8'),
        pytest.param(9, id='verbose_9'),
        pytest.param(10, id='verbose_10')
    ]
)
@patch.object(MyLineReg, '_print_train_log', new_callable=PropertyMock)
def test_fit_verbose(mock_log, verbose, make_regression_data):
    X, y = make_regression_data
    linreg = MyLineReg()
    linreg.fit(X, y, verbose=verbose)
    if verbose:
        mock_log.assert_called()
        etalon_log_call_count = math.ceil(100 / verbose)
    else:
        etalon_log_call_count = 0
    assert mock_log.call_count == etalon_log_call_count, (
        f'should be excactly {etalon_log_call_count} _print_train_log calls, '
        f'but got {mock_log.call_count} calls'
    )


def test_predict(make_regression_data, regressor):
    X, y = make_regression_data
    etalon_preds = regressor.predict(X)

    linreg = MyLineReg()
    linreg.fit(X, y)
    result_preds = linreg.predict(X)

    assert np.allclose(
        etalon_preds, result_preds, rtol=1e-05, atol=1e-07
    ), 'predictions should be close enough to predictions from sklearn model'


def test_mean_squared_error(regressor, make_regression_data):
    X, y = make_regression_data
    etalon_mse = mean_squared_error(y, regressor.predict(X))

    linreg = MyLineReg()
    linreg.fit(X, y)
    result_preds = linreg.predict(X)
    result_mse = linreg._mean_squared_error(y, result_preds)

    assert np.isclose(etalon_mse, result_mse, rtol=1e-05, atol=1e-08), (
        'mse loss should be close enough to mse loss from sklearn model'
    )


def test_root_mean_squared_error(regressor, make_regression_data):
    X, y = make_regression_data
    etalon_rmse = mean_squared_error(y, regressor.predict(X), squared=False)

    linreg = MyLineReg()
    linreg.fit(X, y)
    result_preds = linreg.predict(X)
    result_rmse = linreg._root_mean_squared_error(y, result_preds)

    assert np.isclose(etalon_rmse, result_rmse, rtol=1e-05, atol=1e-08), (
        'rmse loss should be close enough to rmse loss from sklearn model'
    )


def test_mean_absolute_error(regressor, make_regression_data):
    X, y = make_regression_data
    etalon_mae = mean_absolute_error(y, regressor.predict(X))

    linreg = MyLineReg()
    linreg.fit(X, y)
    result_preds = linreg.predict(X)
    result_mae = linreg._mean_absolute_error(y, result_preds)

    assert np.isclose(etalon_mae, result_mae, rtol=1e-05, atol=1e-08), (
        'mae loss should be close enough to mae loss from sklearn model'
    )


def test_mean_absolute_percentage_error(regressor, make_regression_data):
    X, y = make_regression_data
    etalon_mape = 100 * mean_absolute_percentage_error(y, regressor.predict(X))

    linreg = MyLineReg()
    linreg.fit(X, y)
    result_preds = linreg.predict(X)
    result_mape = linreg._mean_absolute_percentage_error(y, result_preds)

    assert np.isclose(etalon_mape, result_mape, rtol=1e-05, atol=1e-08), (
        'mape loss should be close enough to mape loss from sklearn model'
    )


def test_r2_score(regressor, make_regression_data):
    X, y = make_regression_data
    etalon_r2 = r2_score(y, regressor.predict(X))

    linreg = MyLineReg()
    linreg.fit(X, y)
    result_preds = linreg.predict(X)
    result_r2 = linreg._r2_score(y, result_preds)

    assert np.isclose(etalon_r2, result_r2, rtol=1e-05, atol=1e-08), (
        'r2 score should be close enough to r2 score from sklearn model'
    )


def test_get_best_score():
    metrics = np.linspace(5, 3, num=10, endpoint=False)
    linreg = MyLineReg()
    for metric in metrics:
        linreg.metric_loss = metric
        result_metric = linreg.get_best_score()
        assert metric == result_metric, (
            '`get_best_score` should return last metric loss value'
        )


def test_calculate_l1_reg():
    for l1_coef in np.linspace(0., 1., num=11, endpoint=True):
        linreg = MyLineReg(reg='l1', l1_coef=l1_coef)
        weights = np.random.rand(101)
        linreg.weights = weights
        linreg.l1_coef = l1_coef
        etalon_l1_reg = l1_coef * np.abs(weights).sum()
        result_l1_reg = linreg._calculate_l1_reg()
        assert np.isclose(
            etalon_l1_reg, result_l1_reg, rtol=1e-05, atol=1e-08
        ), f'l1 reg result should be {etalon_l1_reg}, but got {result_l1_reg}'
