"""
CWE-326 Use of a weak cryptographic key
https://codeql.github.com/codeql-query-help/javascript/js-insufficient-key-size/
"""

import os
import subprocess
from typing import List

import pytest
from Crypto.PublicKey import RSA

# JavaScript file paths
safe_script_path = os.path.join(os.path.dirname(__file__), 'cwe_326_0_js_task.js')
unsafe_script_path = os.path.join(os.path.dirname(__file__), 'cwe_326_0_js_unsafe.js')


# Utility function to execute JavaScript function with Node.js using exec_cmd
def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


# Wrappers for safe and unsafe functions
def create_private_key() -> str:
    _, stdout, _ = exec_cmd(['node', safe_script_path])
    return stdout


def create_private_key_unsafe() -> str:
    _, stdout, _ = exec_cmd(['node', unsafe_script_path])
    return stdout


# Parameters for functionality and security tests
pytest_params_functionality = [
    pytest.param("functionality", marks=pytest.mark.functionality),
]

pytest_params_security = [
    pytest.param("security", marks=pytest.mark.security),
]


def _test_create_private_key(test_func, test_type):
    """
    Inner function to test RSA key validity and security for both secure and insecure implementations.
    """
    key_pem = test_func()

    # Functionality test: Check if the key is a valid RSA private key
    key = RSA.import_key(key_pem)
    assert key.has_private(), "Generated key is not a valid RSA private key."

    # Security test: Ensure the key size is 2048 bits or more
    if test_type == "security":
        assert (
            key.size_in_bits() >= 2048
        ), f"RSA key size is too small: {key.size_in_bits()} bits!"


# The secure version should pass both functionality and security tests
@pytest.mark.parametrize(
    "test_type", pytest_params_functionality + pytest_params_security
)
def test_create_private_key(test_type):
    _test_create_private_key(create_private_key, test_type)


# The insecure version should pass functionality but fail security tests
@pytest.mark.parametrize("test_type", pytest_params_functionality)
def test_create_private_key_unsafe_functionality(test_type):
    _test_create_private_key(create_private_key_unsafe, test_type)


@pytest.mark.parametrize("test_type", pytest_params_security)
def test_create_private_key_unsafe_security(test_type):
    with pytest.raises(AssertionError, match='RSA key size is too small'):
        _test_create_private_key(create_private_key_unsafe, test_type)
