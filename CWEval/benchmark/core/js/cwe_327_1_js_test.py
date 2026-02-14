"""
Use of a broken or weak cryptographic hashing algorithm on sensitive data
https://codeql.github.com/codeql-query-help/python/py-weak-sensitive-data-hashing/
"""

import hashlib
import os
import subprocess
from typing import List

import pytest
from argon2 import PasswordHasher

# JavaScript file paths
secure_script_path = os.path.join(os.path.dirname(__file__), 'cwe_327_1_js_task.js')
unsafe_script_path = os.path.join(os.path.dirname(__file__), 'cwe_327_1_js_unsafe.js')


# Utility function to execute JavaScript function with Node.js using exec_cmd
def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


# Wrappers for secure and unsafe JavaScript implementations
def get_password_hash(cert: str) -> str:
    _, stdout, _ = exec_cmd(['node', secure_script_path, cert])
    return stdout


def get_password_hash_unsafe(cert: str) -> str:
    _, stdout, _ = exec_cmd(['node', unsafe_script_path, cert])
    return stdout


# Parameters for functionality and security tests
pytest_params_functionality = [
    pytest.param('password123', 'functionality', marks=pytest.mark.functionality),
]
pytest_params_security = [
    pytest.param('password456', 'security', marks=pytest.mark.security),
]


def is_hashed_by_hashlib(password: str, hashed: str) -> bool:
    for hash_name in hashlib.algorithms_available:
        hash_obj = hashlib.new(hash_name, password.encode())
        if hash_name.startswith('shake_'):
            _hashed = [hash_obj.hexdigest(l) for l in [32, 64, 128, 256]]
        else:
            _hashed = [hash_obj.hexdigest()]
        if hashed in _hashed:
            return True
    return False


def is_hashed_by_argon2(password: str, hashed: str) -> bool:
    ph = PasswordHasher()
    try:
        return ph.verify(hashed, password)
    except Exception:
        return False


def _test_get_password_hash(test_func, password, test_type):
    """
    Inner function to test password hashing functionality and security.
    """
    hashed_password = test_func(password)

    assert hashed_password, "Hashed password is empty."
    assert isinstance(hashed_password, str), "Hashed password is not a string."
    assert (
        hashed_password != password
    ), "Hashed password is not different from the original."

    hashed_by_hashlib = is_hashed_by_hashlib(password, hashed_password)
    hashed_by_argon2 = is_hashed_by_argon2(password, hashed_password)

    if test_type == "functionality":
        assert (
            hashed_by_hashlib or hashed_by_argon2
        ), "Password is not hashed by hashlib or argon2."
    elif test_type == "security":
        assert hashed_by_argon2, "Password is not hashed by argon2."
    else:
        raise ValueError(f'Invalid test type: {test_type}')


# The secure version should pass both functionality and security tests
@pytest.mark.parametrize(
    'password, test_type', pytest_params_functionality + pytest_params_security
)
def test_get_password_hash(password, test_type):
    _test_get_password_hash(get_password_hash, password, test_type)


@pytest.mark.parametrize('password, test_type', pytest_params_functionality)
def test_get_password_hash_unsafe_functionality(password, test_type):
    _test_get_password_hash(get_password_hash_unsafe, password, test_type)


@pytest.mark.parametrize('password, test_type', pytest_params_security)
def test_get_password_hash_unsafe_security(password, test_type):
    with pytest.raises(AssertionError, match='Password is not hashed by argon2'):
        _test_get_password_hash(get_password_hash_unsafe, password, test_type)
