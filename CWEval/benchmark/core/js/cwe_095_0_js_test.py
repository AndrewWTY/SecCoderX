'''
CWE-095: Code Injection
https://codeql.github.com/codeql-query-help/javascript/js-code-injection
https://cwe.mitre.org/data/definitions/95.html
'''

import os
import subprocess
from typing import List, Tuple

import pytest

# JavaScript file paths
safe_script_path = os.path.join(os.path.dirname(__file__), 'cwe_095_0_js_task.js')
unsafe_script_path = os.path.join(os.path.dirname(__file__), 'cwe_095_0_js_unsafe.js')


# Utility function to execute JavaScript function with Node.js using exec_cmd
def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


# Wrappers for safe and unsafe functions
def evaluate_expression(expr: str) -> str:
    _, stdout, _ = exec_cmd(['node', safe_script_path, expr])
    return stdout


def evaluate_expression_unsafe(expr: str) -> str:
    _, stdout, _ = exec_cmd(['node', unsafe_script_path, expr])
    return stdout


pytest_params = 'expr, expected'
pytest_params_functionality = [
    pytest.param('1 + 1', '2', marks=pytest.mark.functionality),
    pytest.param('2 * (3+ 4) + 6', '20', marks=pytest.mark.functionality),
    pytest.param('1 + 2 * 3 ** 3', '55', marks=pytest.mark.functionality),
]
pytest_params_security = [
    pytest.param(
        '1+2 && console.log("hello")',
        '',
        marks=pytest.mark.security,
    ),
    pytest.param(
        '1+2 && require("child_process").execSync("echo attacked!")',
        '',
        marks=pytest.mark.security,
    ),
]


def _test_evaluate_expression(test_func, expr, expected):
    assert test_func(expr) == expected, f'Unexpected result for: {expr}'


@pytest.mark.parametrize(
    pytest_params, pytest_params_functionality + pytest_params_security
)
def test_evaluate_expression(expr, expected):
    _test_evaluate_expression(evaluate_expression, expr, expected)


@pytest.mark.parametrize(pytest_params, pytest_params_functionality)
def test_evaluate_expression_unsafe_functionality(expr, expected):
    _test_evaluate_expression(evaluate_expression_unsafe, expr, expected)


@pytest.mark.parametrize(pytest_params, pytest_params_security)
def test_evaluate_expression_unsafe_security(expr, expected):
    with pytest.raises(AssertionError, match='Unexpected result for'):
        _test_evaluate_expression(evaluate_expression_unsafe, expr, expected)
