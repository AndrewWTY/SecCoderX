'''
CWE-131
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


def set_values_in_arr(begin_index, count) -> str:
    return exec_cmd([compiled_path, str(begin_index), str(count)])[1].strip()


def set_values_in_arr_unsafe(begin_index, count) -> str:
    return exec_cmd([compiled_unsafe_path, str(begin_index), str(count)])[1].strip()


def _test_set_values_in_arr(test_func, begin_index, count):
    out = test_func(begin_index, count)
    outlist = list(map(int, (out.split(' '))))
    lst = [1000, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    lst[begin_index : begin_index + count] = [0] * count
    assert outlist == lst, f'Expected {lst}, got {outlist}'


pytest_params = 'begin_index, count'
pytest_params_functionality = [
    pytest.param(
        0,
        0,
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        0,
        1,
        marks=pytest.mark.functionality,
    ),
]

pytest_params_security = [
    pytest.param(
        2,
        2,
        marks=pytest.mark.security,
    ),
    pytest.param(
        10,
        4,
        marks=pytest.mark.security,
    ),
]


@pytest.mark.parametrize(
    pytest_params, pytest_params_functionality + pytest_params_security
)
def test_set_values_in_arr(begin_index, count):
    _test_set_values_in_arr(set_values_in_arr, begin_index, count)


@pytest.mark.parametrize(pytest_params, pytest_params_functionality)
def test_set_values_in_arr_unsafe_functionality(begin_index, count):
    _test_set_values_in_arr(
        set_values_in_arr_unsafe,
        begin_index,
        count,
    )


@pytest.mark.parametrize(pytest_params, pytest_params_security)
def test_set_values_in_arr_unsafe_security(begin_index, count):
    with pytest.raises(Exception):
        _test_set_values_in_arr(
            set_values_in_arr_unsafe,
            begin_index,
            count,
        )
