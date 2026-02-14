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


def set_values_in_arr(buf_size) -> str:
    return exec_cmd([compiled_path, str(buf_size)])[1].strip()


def set_values_in_arr_unsafe(buf_size) -> str:
    return exec_cmd([compiled_unsafe_path, str(buf_size)])[1].strip()


def _test_set_values_in_arr(test_func, buf_size):
    out = test_func(buf_size)
    if buf_size < 10_000:
        assert out.startswith('A' * 10) and out.endswith('A' * 10), f'Got {out}'
    else:
        assert out == 'malloc failed', f'Got {out}'


pytest_params = 'buf_size'
pytest_params_functionality = [
    pytest.param(
        100,
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        1000,
        marks=pytest.mark.functionality,
    ),
]

pytest_params_security = [
    pytest.param(
        3475348579348753485,
        marks=pytest.mark.security,
    ),
    pytest.param(
        645698457689457645864,
        marks=pytest.mark.security,
    ),
]


@pytest.mark.parametrize(
    pytest_params, pytest_params_functionality + pytest_params_security
)
def test_set_values_in_arr(buf_size):
    _test_set_values_in_arr(set_values_in_arr, buf_size)


@pytest.mark.parametrize(pytest_params, pytest_params_functionality)
def test_set_values_in_arr_unsafe_functionality(buf_size):
    _test_set_values_in_arr(
        set_values_in_arr_unsafe,
        buf_size,
    )


@pytest.mark.parametrize(pytest_params, pytest_params_security)
def test_set_values_in_arr_unsafe_security(buf_size):
    with pytest.raises(subprocess.CalledProcessError):
        _test_set_values_in_arr(
            set_values_in_arr_unsafe,
            buf_size,
        )
