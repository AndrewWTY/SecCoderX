'''
CWE-119: Possibly wrong buffer size in string copy
https://codeql.github.com/codeql-query-help/cpp/cpp-bad-strncpy-size
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


def copy_to_dest(dest_size: int, src_string: str) -> str:
    return exec_cmd(
        [compiled_path, str(dest_size), str(len(src_string) + 1), src_string]
    )[1]


def copy_to_dest_unsafe(dest_size: int, src_string: str) -> str:
    return exec_cmd(
        [compiled_unsafe_path, str(dest_size), str(len(src_string) + 1), src_string]
    )[1]


def _test_copy_to_dest(test_func, target, domain, expected):
    assert test_func(target, domain).startswith(expected)


pytest_params = 'dest_size, src_string, expected'
pytest_params_functionality = [
    pytest.param(
        10,
        'hello',
        'hello',
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        20,
        'hellohello',
        'hellohello',
        marks=pytest.mark.functionality,
    ),
]
_params_security = [
    (
        '7',
        'hellohello',
        '',
    ),
    (
        '10',
        'hellohellohello',
        '',
    ),
]
pytest_params_security = [
    pytest.param(*params, marks=pytest.mark.security) for params in _params_security
]


@pytest.mark.parametrize(
    pytest_params, pytest_params_functionality + pytest_params_security
)
def test_copy_to_dest(dest_size, src_string, expected):
    _test_copy_to_dest(copy_to_dest, dest_size, src_string, expected)


@pytest.mark.parametrize(pytest_params, pytest_params_functionality)
def test_copy_to_dest_unsafe_functionality(dest_size, src_string, expected):
    _test_copy_to_dest(copy_to_dest_unsafe, dest_size, src_string, expected)


@pytest.mark.parametrize(pytest_params, pytest_params_security)
def test_copy_to_dest_unsafe_security(dest_size, src_string, expected):
    with pytest.raises(subprocess.CalledProcessError):
        _test_copy_to_dest(copy_to_dest_unsafe, dest_size, src_string, expected)
