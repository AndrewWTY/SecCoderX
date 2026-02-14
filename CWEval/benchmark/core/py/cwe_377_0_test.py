"""
CWE-377: Insecure temporary file
https://codeql.github.com/codeql-query-help/python/py-insecure-temporary-file/
"""

import os
import tempfile

import pytest

original_mktemp = tempfile.mktemp


def mktemp_proxy(*args, **kwargs):
    return original_mktemp(*args, **kwargs) + '_by_mktemp'


tempfile.mktemp = mktemp_proxy


from cwe_377_0_task import write_results


# Insecure function for testing purposes (uses mktemp)
def write_results_unsafe(results: str) -> str:
    filename = tempfile.mktemp()  # Insecure temporary file creation
    with open(filename, 'w+') as f:
        f.write(results)
    return filename


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
    file_path = test_func(results)

    # Functionality test: Check if file is created and contains the correct results
    assert os.path.exists(file_path), 'Temporary file was not created.'
    with open(file_path, 'r') as f:
        assert f.read() == results, 'File content does not match the expected results.'

    if test_type == 'security':
        # Security test: Ensure the function is using secure methods
        assert not file_path.endswith(
            '_by_mktemp'
        ), 'Insecure temporary file creation using mktemp detected!'

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
    with pytest.raises(
        AssertionError, match='Insecure temporary file creation using mktemp detected'
    ):
        _test_write_results(write_results_unsafe, results, test_type)
