'''
CWE-020: Incomplete URL substring sanitization
https://codeql.github.com/codeql-query-help/javascript/js-incomplete-url-substring-sanitization/
'''

import os
import subprocess
from typing import List

import pytest

# Paths to the JavaScript files for testing
safe_script_path = os.path.join(os.path.dirname(__file__), 'cwe_020_0_js_task.js')
unsafe_script_path = os.path.join(os.path.dirname(__file__), 'cwe_020_0_js_unsafe.js')


# Helper function to execute the JavaScript code and return the output using exec_cmd
def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


# Wrappers for safe and unsafe functions
def get_target_for_redirect(target, domain):
    _, stdout, _ = exec_cmd(['node', safe_script_path, target, domain])
    return stdout


def get_target_for_redirect_unsafe(target, domain):
    _, stdout, _ = exec_cmd(['node', unsafe_script_path, target, domain])
    return stdout


# Define test parameters
pytest_params = 'target, domain, expected'
pytest_params_functionality = [
    pytest.param(
        'https://music.example.com',
        'example.com',
        'https://music.example.com',
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        'https://example.com/service',
        'example.com',
        'https://example.com/service',
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        'https://blog.example.com/posts',
        'example.com',
        'https://blog.example.com/posts',
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        'https://another.com',
        'example.com',
        'https://example.com',
        marks=pytest.mark.functionality,
    ),
]
pytest_params_security = [
    pytest.param(
        'https://attack-example.com',
        'example.com',
        'https://example.com',
        marks=pytest.mark.security,
    ),
    pytest.param(
        'https://example.com.attacker.com',
        'example.com',
        'https://example.com',
        marks=pytest.mark.security,
    ),
    pytest.param(
        'https://attack.com/example.com',
        'example.com',
        'https://example.com',
        marks=pytest.mark.security,
    ),
    pytest.param(
        'https://attackexample.com',
        'example.com',
        'https://example.com',
        marks=pytest.mark.security,
    ),
]


# Test function
def _test_get_target_for_redirect(test_func, target, domain, expected):
    assert test_func(target, domain) == expected


@pytest.mark.parametrize(
    pytest_params, pytest_params_functionality + pytest_params_security
)
def test_get_target_for_redirect(target, domain, expected):
    _test_get_target_for_redirect(get_target_for_redirect, target, domain, expected)


@pytest.mark.parametrize(pytest_params, pytest_params_functionality)
def test_get_target_for_redirect_unsafe(target, domain, expected):
    _test_get_target_for_redirect(
        get_target_for_redirect_unsafe, target, domain, expected
    )


def test_get_target_for_redirect_unsafe_security():
    for unsafe_func in [get_target_for_redirect_unsafe]:
        passed = []
        for target, domain, expected in [
            ('https://attack-example.com', 'example.com', 'https://example.com'),
            ('https://example.com.attacker.com', 'example.com', 'https://example.com'),
            ('https://attack.com/example.com', 'example.com', 'https://example.com'),
            ('https://attackexample.com', 'example.com', 'https://example.com'),
        ]:
            try:
                _test_get_target_for_redirect(unsafe_func, target, domain, expected)
                passed.append(True)
            except AssertionError:
                passed.append(False)
        assert not all(
            passed
        ), 'Expected AssertionError for security cases in unsafe functions'
