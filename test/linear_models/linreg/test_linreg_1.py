from mock import patch
import pytest

from src.linear_models.linreg.linreg_1 import MyLineReg


def test_can_import_class():
    from src.linear_models.linreg.linreg_1 import MyLineReg


@pytest.mark.parametrize(
        'n_iter, learning_rate, default_n_iter, default_lr',
        [
            pytest.param(None, None, 100, 0.1, id='all_defaults'),
            pytest.param(10, None, 100, 0.1, id='set_n_iter'),
            pytest.param(None, 0.001, 100, 0.1, id='set_learning_rate'),
            pytest.param(10, 0.001, 100, 0.1, id='set_all_params'),
        ]
)
def test_class_initialization(n_iter, learning_rate, default_n_iter, default_lr):
    linreg = MyLineReg(
        n_iter if n_iter else default_n_iter,
        learning_rate if learning_rate else default_lr
    )
    etalon_n_iter = n_iter if n_iter else default_n_iter
    etalon_learning_rate = learning_rate if learning_rate else default_lr
    assert linreg.n_iter == etalon_n_iter, (
        f'n_iter should be {etalon_n_iter}, but got {linreg.n_iter}'
    )
    assert linreg.learning_rate == etalon_learning_rate, (
        f'n_iter should be {etalon_learning_rate}, but got {linreg.learning_rate}'
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
def test_class_representation(n_iter, learning_rate, default_n_iter, default_lr):
    representation_string = 'MyLineReg class: n_iter={n_iter}, learning_rate={learning_rate}'
    linreg = MyLineReg(
        n_iter if n_iter else default_n_iter,
        learning_rate if learning_rate else default_lr
    )
    print(linreg)
    etalon_n_iter = n_iter if n_iter else default_n_iter
    etalon_learning_rate = learning_rate if learning_rate else default_lr
    etalon_representation = representation_string.format(
        n_iter=etalon_n_iter, learning_rate=etalon_learning_rate
    )
    assert linreg.__repr__() == etalon_representation, (
        f'class representation should be {etalon_representation} '
        f'but got {linreg.__repr__()}'
    )
