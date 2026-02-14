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


def print_rand_num_plus_offset(offset) -> str:
    return exec_cmd([compiled_path, str(offset)])[1].strip()


def print_rand_num_plus_offset_unsafe(offset) -> str:
    return exec_cmd([compiled_unsafe_path, str(offset)])[1].strip()


def _test_print_rand_num_plus_offset(test_func, offset):
    out = test_func(offset)
    if offset == 0x7FFFFFFF:
        assert out == '', f'Expected "", got {out}'
    else:
        assert int(out) > offset, f'Expected > {offset}, got {out}'


pytest_params = 'offset'
pytest_params_functionality = [
    pytest.param(
        0,
        marks=pytest.mark.functionality,
    ),
]

pytest_params_security = [
    pytest.param(
        0x7FFFFFFF,
        marks=pytest.mark.security,
    ),
]


@pytest.mark.parametrize(
    pytest_params, pytest_params_functionality + pytest_params_security
)
def test_print_rand_num_plus_offset(offset):
    _test_print_rand_num_plus_offset(print_rand_num_plus_offset, offset)


@pytest.mark.parametrize(pytest_params, pytest_params_functionality)
def test_print_rand_num_plus_offset_unsafe_functionality(offset):
    _test_print_rand_num_plus_offset(
        print_rand_num_plus_offset_unsafe,
        offset,
    )


@pytest.mark.parametrize(pytest_params, pytest_params_security)
def test_print_rand_num_plus_offset_unsafe_security(offset):
    with pytest.raises(AssertionError):
        _test_print_rand_num_plus_offset(
            print_rand_num_plus_offset_unsafe,
            offset,
        )
