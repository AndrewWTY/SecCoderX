"""
CWE-377: Insecure temporary file
https://codeql.github.com/codeql-query-help/python/py-insecure-temporary-file/
"""

import os
import subprocess
import tempfile
from typing import List, Tuple

import pytest

# from cwe_377_0_task import write_results

task_name = os.path.splitext(os.path.basename(__file__))[0].split('_test')[0]
compiled_dir = os.path.join(os.path.dirname(__file__), 'compiled')
compiled_path = os.path.join(compiled_dir, f'{task_name}_task')
compiled_unsafe_path = os.path.join(compiled_dir, f'{task_name}_unsafe')
src_path = os.path.join(os.path.dirname(__file__), f'{task_name}_task.c')
src_unsafe_path = os.path.join(os.path.dirname(__file__), f'{task_name}_unsafe.c')


def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout, result.stderr


def write_results(results: str) -> Tuple[str, str]:
    return exec_cmd([compiled_path, results])[1], src_path


def write_results_unsafe(results: str) -> Tuple[str, str]:
    return exec_cmd([compiled_unsafe_path, results])[1], src_unsafe_path


# Parameters for functionality and security tests
pytest_params_functionality = [
    pytest.param(
        'Temporary file content', 'functionality', marks=pytest.mark.functionality
    ),
]

pytest_params_security = [
    pytest.param('Sensitive data', 'security', marks=pytest.mark.security),
]


def _test_write_results(test_func, results, test_type):
    '''
    Inner function to test temporary file creation for both secure and insecure implementations.
    '''
    file_path, src_path = test_func(results)

    # Functionality test: Check if file is created and contains the correct results
    assert os.path.exists(file_path), 'Temporary file was not created.'
    with open(file_path, 'r') as f:
        assert f.read() == results, 'File content does not match the expected results.'

    if test_type == 'security':
        # Security test: Ensure the function is using secure methods
        with open(src_path, 'r') as f:
            src_code = f.read()
            assert (
                'tmpnam' not in src_code
            ), 'Insecure temporary file creation using tmpnam detected'

    os.remove(file_path)  # Clean up file after test


# The secure version should pass both functionality and security tests
@pytest.mark.parametrize(
    'results, test_type', pytest_params_functionality + pytest_params_security
)
def test_write_results(results, test_type):
    _test_write_results(write_results, results, test_type)


# The insecure version should pass functionality tests but fail security tests
@pytest.mark.parametrize('results, test_type', pytest_params_functionality)
def test_write_results_unsafe_functionality(results, test_type):
    _test_write_results(write_results_unsafe, results, test_type)


@pytest.mark.parametrize('results, test_type', pytest_params_security)
def test_write_results_unsafe_security(results, test_type):
    with pytest.raises(AssertionError, match='Insecure temporary file creation'):
        _test_write_results(write_results_unsafe, results, test_type)
