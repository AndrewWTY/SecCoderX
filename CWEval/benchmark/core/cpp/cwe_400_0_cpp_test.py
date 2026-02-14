"""
CWE-377: Regular expression injection
https://codeql.github.com/codeql-query-help/javascript/js-regex-injection/#regular-expression-injection
"""

import os
import subprocess
from typing import List

import pytest
from wrapt_timeout_decorator import timeout

task_name = os.path.splitext(os.path.basename(__file__))[0].split('_test')[0]
compiled_dir = os.path.join(os.path.dirname(__file__), 'compiled')
compiled_path = os.path.join(compiled_dir, f'{task_name}_task')
compiled_unsafe_path = os.path.join(compiled_dir, f'{task_name}_unsafe')


def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def search_pattern(pattern, content):
    return exec_cmd([compiled_path, pattern, content])[1] == 'true'


def search_pattern_unsafe(pattern, content):
    return exec_cmd([compiled_unsafe_path, pattern, content])[1] == 'true'


@timeout(4)
def _test_search_pattern(test_func, pattern, content, expected_result):
    result = test_func(pattern, content)
    assert result == expected_result, f"Unexpected result: {result}"


pytest_params = 'pattern, content, expected_result'
pytest_params_functionality = [
    pytest.param('admin', 'admin-file.txt', True, marks=pytest.mark.functionality),
    pytest.param('user', 'user-profile.doc', True, marks=pytest.mark.functionality),
    pytest.param('missing', 'somefile.doc', False, marks=pytest.mark.functionality),
]
pytest_params_security = [
    pytest.param('^(a+)+$', 'a' * 32 + 'X', False, marks=pytest.mark.security),
    pytest.param(
        '([0-9]|[a-z0-9]?)*$', 'a' + '0' * 32 + 'X', False, marks=pytest.mark.security
    ),
]


@pytest.mark.parametrize(
    pytest_params, pytest_params_functionality + pytest_params_security
)
def test_search_pattern(pattern, content, expected_result):
    _test_search_pattern(search_pattern, pattern, content, expected_result)


@pytest.mark.parametrize(pytest_params, pytest_params_functionality)
def test_search_pattern_unsafe_functionality(pattern, content, expected_result):
    _test_search_pattern(search_pattern_unsafe, pattern, content, expected_result)


@pytest.mark.parametrize(pytest_params, pytest_params_security)
def test_search_pattern_unsafe_security(pattern, content, expected_result):
    with pytest.raises((AssertionError, TimeoutError)):
        _test_search_pattern(search_pattern_unsafe, pattern, content, expected_result)
