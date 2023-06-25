from mock import patch
from mock.mock import PropertyMock
import pytest

from src.linear_models.linreg.linreg_1 import MyLineReg


def test_can_import_class():
    from src.linear_models.linreg.linreg_1 import MyLineReg


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
def test_class_representation(mock_repr, n_iter, learning_rate, default_n_iter, default_lr):
    etalon_n_iter = n_iter if n_iter else default_n_iter
    etalon_learning_rate = learning_rate if learning_rate else default_lr
    representation_string = '{class_name}: n_iter={n_iter}, lr={learning_rate}'
    mock_repr.return_value = representation_string
    linreg = MyLineReg(etalon_n_iter, etalon_learning_rate)
    result = repr(linreg)
    mock_repr.assert_called()
    etalon_representation = representation_string.format(
        class_name = linreg.__class__.__name__,
        n_iter=etalon_n_iter,
        learning_rate=etalon_learning_rate
    )

    assert result == etalon_representation, (
        f'class representation should be {etalon_representation} '
        f'but got {linreg.__repr__()}'
    )
