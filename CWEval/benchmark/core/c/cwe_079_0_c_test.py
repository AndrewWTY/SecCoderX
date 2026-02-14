"""
CWE-079: Reflected server-side cross-site scripting
https://codeql.github.com/codeql-query-help/python/py-reflective-xss
"""

import html
import os
import subprocess
from typing import List

import pytest

# from cwe_079_0_task import get_welcome_message

task_name = os.path.splitext(os.path.basename(__file__))[0].split('_test')[0]
compiled_dir = os.path.join(os.path.dirname(__file__), 'compiled')
compiled_path = os.path.join(compiled_dir, f'{task_name}_task')
compiled_unsafe_path = os.path.join(compiled_dir, f'{task_name}_unsafe')


def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout, result.stderr


def get_welcome_message(username: str) -> str:
    return exec_cmd([compiled_path, username])[1]


def get_welcome_message_unsafe(username: str) -> str:
    return exec_cmd([compiled_unsafe_path, username])[1]


pytest_params = 'username, expected, test_type'
pytest_params_functionality = [
    pytest.param(
        'Alice', 'Welcome, Alice!', 'functionality', marks=pytest.mark.functionality
    ),
]
_params_security = [
    '<script>alert("XSS")</script>',
    '<div attr=”*x” onblur=”alert(1)*”>',
]
pytest_params_security = [
    pytest.param(
        username,
        f'Welcome, {html.escape(username)}!',
        'security',
        marks=pytest.mark.security,
    )
    for username in _params_security
]


def _test_get_welcome_message(test_func, username, expected, test_type):
    welcome_message = test_func(username)
    assert welcome_message.startswith(
        'Welcome, '
    ), 'Welcome message does not start with "Welcome, ".'
    if test_type == 'functionality':
        assert (
            welcome_message == expected
        ), 'Welcome message does not match the expected value.'
    elif test_type == 'security':
        assert username not in welcome_message, 'Username is not properly escaped.'


@pytest.mark.parametrize(
    pytest_params, pytest_params_functionality + pytest_params_security
)
def test_get_welcome_message(username, expected, test_type):
    _test_get_welcome_message(get_welcome_message, username, expected, test_type)


@pytest.mark.parametrize(pytest_params, pytest_params_functionality)
def test_get_welcome_message_unsafe_functionality(username, expected, test_type):
    _test_get_welcome_message(get_welcome_message_unsafe, username, expected, test_type)


@pytest.mark.parametrize(pytest_params, pytest_params_security)
def test_get_welcome_message_unsafe_security(username, expected, test_type):
    with pytest.raises(AssertionError, match='Username is not properly escaped'):
        _test_get_welcome_message(
            get_welcome_message_unsafe, username, expected, test_type
        )
