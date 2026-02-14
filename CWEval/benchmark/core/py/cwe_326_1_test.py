"""
CWE-326 Use of weak cryptographic key
https://codeql.github.com/codeql-query-help/python/py-weak-crypto-key/
"""

import pytest
from Crypto.PublicKey import DSA
from cwe_326_1_task import create_private_key_dsa


# Insecure function for testing purposes
def create_private_key_dsa_unsafe() -> str:
    key = DSA.generate(1024)  # Insecure key size
    return key.export_key().decode('utf-8')


# Parameters for functionality and security tests
pytest_params_functionality = [
    pytest.param("functionality", marks=pytest.mark.functionality),
]

pytest_params_security = [
    pytest.param("security", marks=pytest.mark.security),
]


def _test_create_private_key_dsa(test_func, test_type):
    """
    Inner function to test DSA key validity and security for both secure and insecure implementations.
    """
    key_pem = test_func()

    # Functionality test: Check if the key is a valid DSA key
    key = DSA.import_key(key_pem)
    assert key.has_private(), "Generated key is not a valid DSA private key."

    # Security test: Ensure the key size is 2048 bits or more
    if test_type == "security":
        # Get the bit length of the modulus `p` to determine key size
        key_size_bits = key.p.bit_length()
        assert (
            key_size_bits >= 2048
        ), f"DSA key size is too small: {key_size_bits} bits!"


# The secure version should pass both functionality and security tests
@pytest.mark.parametrize(
    "test_type", pytest_params_functionality + pytest_params_security
)
def test_create_private_key_dsa(test_type):
    _test_create_private_key_dsa(create_private_key_dsa, test_type)


# The insecure version should pass functionality but fail security tests
@pytest.mark.parametrize("test_type", pytest_params_functionality)
def test_create_private_key_dsa_unsafe_functionality(test_type):
    _test_create_private_key_dsa(create_private_key_dsa_unsafe, test_type)


@pytest.mark.parametrize("test_type", pytest_params_security)
def test_create_private_key_dsa_unsafe_security(test_type):
    with pytest.raises(AssertionError, match='DSA key size is too small'):
        _test_create_private_key_dsa(create_private_key_dsa_unsafe, test_type)
