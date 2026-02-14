"""
CWE-326 Use of weak cryptographic key
https://codeql.github.com/codeql-query-help/python/py-weak-crypto-key/
"""

import pytest
from Crypto.PublicKey import RSA
from cwe_326_0_task import create_private_key


# Insecure function for testing purposes
def create_private_key_unsafe() -> str:
    key = RSA.generate(1024)  # Insecure key size
    return key.export_key().decode('utf-8')


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

    # Functionality test: Check if the key is a valid RSA key
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
