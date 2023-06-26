from mock import patch  # type: ignore
from mock.mock import PropertyMock  # type: ignore
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
import pytest

from src.linear_models.linreg.linreg_2 import MyLineReg


def test_can_import_class():
    from src.linear_models.linreg.linreg_2 import MyLineReg  # noqa: F401


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


def test_add_bias():
    features, _ = make_regression(
        n_samples=1000,
        n_features=14,
        n_informative=10,
        noise=15,
        random_state=42
    )
    etalon_features = features.copy()
    features = pd.DataFrame(features)
    etalon_features = pd.DataFrame(etalon_features)
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


def test_initialize_weights():
    features, _ = make_regression(
        n_samples=1000,
        n_features=14,
        n_informative=10,
        noise=15,
        random_state=42
    )
    features = pd.DataFrame(features)

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
