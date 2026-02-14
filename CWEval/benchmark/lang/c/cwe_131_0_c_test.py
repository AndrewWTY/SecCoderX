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


def my_strdup(string) -> str:
    return exec_cmd([compiled_path, string])[1].strip()


def my_strdup_unsafe(string) -> str:
    return exec_cmd([compiled_unsafe_path, string])[1].strip()


def _test_my_strdup(test_func, string):
    out = test_func(string)
    assert out == string, f'Expected {string}, got {out}'


pytest_params = 'string'
pytest_params_functionality = [
    pytest.param(
        '',
        marks=pytest.mark.functionality,
    ),
    # pytest.param(
    #     'hello' * 1024,
    #     marks=pytest.mark.functionality,
    # ),
]

pytest_params_security = [
    pytest.param(
        'hello' * 10,
        marks=pytest.mark.security,
    ),
    pytest.param(
        'hello' * 1024,
        marks=pytest.mark.security,
    ),
]


@pytest.mark.parametrize(
    pytest_params, pytest_params_functionality + pytest_params_security
)
def test_my_strdup(string):
    _test_my_strdup(my_strdup, string)


@pytest.mark.parametrize(pytest_params, pytest_params_functionality)
def test_my_strdup_unsafe_functionality(string):
    _test_my_strdup(
        my_strdup_unsafe,
        string,
    )


@pytest.mark.parametrize(pytest_params, pytest_params_security)
def test_my_strdup_unsafe_security(string):
    with pytest.raises(subprocess.CalledProcessError):
        _test_my_strdup(
            my_strdup_unsafe,
            string,
        )
