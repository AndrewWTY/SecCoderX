"""
CWE-079: Reflected cross-site scripting
https://codeql.github.com/codeql-query-help/javascript/js-reflected-xss/
"""

import html
import os
import subprocess
from typing import List

import pytest

# JavaScript file paths
safe_script_path = os.path.join(os.path.dirname(__file__), 'cwe_079_0_js_task.js')
unsafe_script_path = os.path.join(os.path.dirname(__file__), 'cwe_079_0_js_unsafe.js')


# Utility function to execute JavaScript function with Node.js using exec_cmd
def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


# Wrappers for safe and unsafe functions
def get_welcome_message(username):
    _, stdout, _ = exec_cmd(['node', safe_script_path, username])
    return stdout


def get_welcome_message_unsafe(username):
    _, stdout, _ = exec_cmd(['node', unsafe_script_path, username])
    return stdout


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
