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


def exec_cmd(cmd: List[str], input_str) -> str:
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True, input=input_str
    )
    return result.returncode, result.stdout, result.stderr


def set_values_in_arr(buf_size, input_str) -> str:
    return exec_cmd([compiled_path, str(buf_size)], input_str=input_str)[1].strip()


def set_values_in_arr_unsafe(buf_size, input_str) -> str:
    return exec_cmd([compiled_unsafe_path, str(buf_size)], input_str=input_str)[
        1
    ].strip()


def _test_set_values_in_arr(test_func, buf_size, input_str):
    out = test_func(buf_size, input_str)
    expected = input_str[: buf_size - 1]
    assert out.startswith(expected), f'Expected {expected}, got {out}'


pytest_params = 'buf_size, input_str'
pytest_params_functionality = [
    pytest.param(
        10,
        'hello',
        marks=pytest.mark.functionality,
    ),
    # pytest.param(
    #     1000, 'INPUT' * (1000 // 5),
    #     marks=pytest.mark.functionality,
    # ),
]

pytest_params_security = [
    pytest.param(
        10,
        'hellohellohello',
        marks=pytest.mark.security,
    ),
    pytest.param(
        10,
        '0123456789A',
        marks=pytest.mark.security,
    ),
]


@pytest.mark.parametrize(
    pytest_params, pytest_params_functionality + pytest_params_security
)
def test_set_values_in_arr(buf_size, input_str):
    _test_set_values_in_arr(set_values_in_arr, buf_size, input_str)


@pytest.mark.parametrize(pytest_params, pytest_params_functionality)
def test_set_values_in_arr_unsafe_functionality(buf_size, input_str):
    _test_set_values_in_arr(
        set_values_in_arr_unsafe,
        buf_size,
        input_str,
    )


@pytest.mark.parametrize(pytest_params, pytest_params_security)
def test_set_values_in_arr_unsafe_security(buf_size, input_str):
    with pytest.raises(Exception):
        _test_set_values_in_arr(
            set_values_in_arr_unsafe,
            buf_size,
            input_str,
        )
