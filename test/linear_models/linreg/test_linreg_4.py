from mock import patch  # type: ignore
from mock.mock import PropertyMock  # type: ignore
import logging
import math
import pytest

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

from src.linear_models.linreg.linreg_4 import MyLineReg, APPLICATION_NAME


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
    from src.linear_models.linreg.linreg_4 import MyLineReg  # noqa: F401


@pytest.mark.parametrize(
        'n_iter, learning_rate, etalon_n_iter, etalon_lr',
        [
            pytest.param(None, None, 100, 0.1, id='all_defaults'),
            pytest.param(10, None, 10, 0.1, id='set_n_iter'),
            pytest.param(None, 0.001, 100, 0.001, id='set_learning_rate'),
            pytest.param(10, 0.001, 10, 0.001, id='set_all_params'),
        ]
)
def test_class_initialization(n_iter, learning_rate, etalon_n_iter, etalon_lr):
    if n_iter and learning_rate:
        linreg = MyLineReg(n_iter, learning_rate)
    elif n_iter:
        linreg = MyLineReg(n_iter)
    elif learning_rate:
        linreg = MyLineReg(learning_rate=learning_rate)
    else:
        linreg = MyLineReg()
    assert linreg.n_iter == etalon_n_iter, (
        f'n_iter should be {etalon_n_iter}, but got {linreg.n_iter}'
    )
    assert linreg.learning_rate == etalon_lr, (
        f'n_iter should be {etalon_lr}, but got {linreg.learning_rate}'
    )


@pytest.mark.parametrize(
        'n_iter, learning_rate, default_n_iter, default_lr',
        [
            pytest.param(None, None, 100, 0.1, id='all_defaults'),
            pytest.param(10, None, 100, 0.1, id='set_n_iter'),
            pytest.param(None, 0.001, 100, 0.1, id='set_learning_rate'),
            pytest.param(10, 0.001, 100, 0.1, id='set_all_params'),
        ]
)
@patch.object(MyLineReg, 'REPR_STR', new_callable=PropertyMock)
def test_class_representation(
    mock_repr, n_iter, learning_rate, default_n_iter, default_lr
):
    etalon_n_iter = n_iter if n_iter else default_n_iter
    etalon_learning_rate = learning_rate if learning_rate else default_lr
    representation_string = '{class_name}: n_iter={n_iter}, lr={learning_rate}'
    mock_repr.return_value = representation_string
    linreg = MyLineReg(etalon_n_iter, etalon_learning_rate)
    result = repr(linreg)
    mock_repr.assert_called()
    etalon_representation = representation_string.format(
        class_name=linreg.__class__.__name__,
        n_iter=etalon_n_iter,
        learning_rate=etalon_learning_rate
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
            start=i if i else 'start', loss=loss
        )
        assert any(
            etalon_log_string in message for message in caplog.messages
        ), f'log string `{etalon_log_string}` should be in {caplog.messages},'
    assert mock_log.call_count == iters.size, (
        f'should be excectly {iters.size}, but got {mock_log.call_count}'
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
