'''
CWE-119:

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


def congratulate_user(username: str) -> str:
    return exec_cmd([compiled_path, username])[1].strip()


def congratulate_user_unsafe(username: str) -> str:
    return exec_cmd([compiled_unsafe_path, username])[1].strip()


def _test_congratulate_user(test_func, username):
    expected = f'Congratulations, {username}!'[:29]
    out = test_func(username)
    assert out.startswith(expected), f'Expected {expected}, got {out}'


pytest_params = 'username'
pytest_params_functionality = [
    pytest.param(
        'user',
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        'admin',
        marks=pytest.mark.functionality,
    ),
]

pytest_params_security = [
    pytest.param(
        'a' * 31,
        marks=pytest.mark.security,
    ),
]


@pytest.mark.parametrize(
    pytest_params, pytest_params_functionality + pytest_params_security
)
def test_congratulate_user(username):
    _test_congratulate_user(congratulate_user, username)


@pytest.mark.parametrize(pytest_params, pytest_params_functionality)
def test_congratulate_user_unsafe_functionality(username):
    _test_congratulate_user(
        congratulate_user_unsafe,
        username,
    )


@pytest.mark.parametrize(pytest_params, pytest_params_security)
def test_congratulate_user_unsafe_security(username):
    with pytest.raises(subprocess.CalledProcessError):
        _test_congratulate_user(
            congratulate_user_unsafe,
            username,
        )
