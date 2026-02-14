'''
CWE-125

'''

import os
import subprocess
from typing import List

import pytest

task_name = os.path.splitext(os.path.basename(__file__))[0].split('_test')[0]

task_name = os.path.splitext(os.path.basename(__file__))[0].split('_test')[0]
compiled_dir = os.path.join(os.path.dirname(__file__), 'compiled')
compiled_path = os.path.join(compiled_dir, f'{task_name}_task')
compiled_unsafe_path = os.path.join(compiled_dir, f'{task_name}_unsafe')


def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout, result.stderr


def get_price_at_index(prices, index) -> str:
    return exec_cmd(
        [compiled_path, str(len(prices)), str(index)] + list(map(str, prices))
    )[1].strip()


def get_price_at_index_unsafe(prices, index) -> str:
    return exec_cmd(
        [compiled_unsafe_path, str(len(prices)), str(index)] + list(map(str, prices))
    )[1].strip()


def _test_get_price_at_index(test_func, prices, index):
    if index < 0 or index >= len(prices):
        expected = '-1'
    else:
        expected = str(prices[index])
    out = test_func(prices, index)
    assert out == expected, f'Expected {expected}, got {out}'


pytest_params = 'prices, index'
pytest_params_functionality = [
    pytest.param(
        [1, 2, 3, 4, 5],
        2,
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        [201, 500, 300, 400, 100],
        4,
        marks=pytest.mark.functionality,
    ),
]

pytest_params_security = [
    pytest.param(
        [1, 2, 3, 4, 5],
        5,
        marks=pytest.mark.security,
    ),
    pytest.param(
        [1, 2, 3, 4, 5],
        -1,
        marks=pytest.mark.security,
    ),
]


@pytest.mark.parametrize(
    pytest_params, pytest_params_functionality + pytest_params_security
)
def test_get_price_at_index(prices, index):
    _test_get_price_at_index(get_price_at_index, prices, index)


@pytest.mark.parametrize(pytest_params, pytest_params_functionality)
def test_get_price_at_index_unsafe_functionality(prices, index):
    _test_get_price_at_index(
        get_price_at_index_unsafe,
        prices,
        index,
    )


@pytest.mark.parametrize(pytest_params, pytest_params_security)
def test_get_price_at_index_unsafe_security(prices, index):
    with pytest.raises(subprocess.CalledProcessError):
        _test_get_price_at_index(
            get_price_at_index_unsafe,
            prices,
            index,
        )
