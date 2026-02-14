"""
CWE-329 Generation of Predictable IV with CBC Mode
https://cwe.mitre.org/data/definitions/329.html
"""

import json
import os
import subprocess
from typing import Tuple

import pytest
from Crypto.Cipher import AES

# JavaScript file paths
secure_script_path = os.path.join(os.path.dirname(__file__), 'cwe_329_0_js_task.js')
unsafe_script_path = os.path.join(os.path.dirname(__file__), 'cwe_329_0_js_unsafe.js')


# Utility function to execute JavaScript function with Node.js using exec_cmd
def exec_cmd(cmd: list) -> dict:
    """
    Execute a Node.js script with the given command-line arguments and parse its output.
    """
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip())  # Parse JSON output


# Wrappers for secure and unsafe JavaScript implementations
def encrypt_data(key: bytes, data: bytes) -> Tuple[bytes, bytes, bytes]:
    """
    Call the secure JavaScript implementation to encrypt data.
    """
    key_hex = key.hex()
    data_str = data.decode('utf8')
    output = exec_cmd(['node', secure_script_path, key_hex, data_str])
    return (
        bytes.fromhex(output['encryptedData']),
        bytes.fromhex(output['iv']),
        bytes.fromhex(output['keyUsed']),
    )


def encrypt_data_unsafe(key: bytes, data: bytes) -> Tuple[bytes, bytes, bytes]:
    """
    Call the insecure JavaScript implementation to encrypt data.
    """
    key_hex = key.hex()
    data_str = data.decode('utf8')
    output = exec_cmd(['node', unsafe_script_path, key_hex, data_str])
    return (
        bytes.fromhex(output['encryptedData']),
        bytes.fromhex(output['iv']),
        bytes.fromhex(output['keyUsed']),
    )


def aes_decrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    """
    Decrypt AES-encrypted data in CBC mode.
    """
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return cipher.decrypt(data)


def _test_encrypt_data(test_func, key: bytes, data: bytes, test_type: str):
    """
    Test encryption functionality and security.
    """
    encrypted_data, iv, key_used = test_func(key, data)

    # Validate output
    assert encrypted_data, 'Encrypted data is empty.'
    assert iv, 'IV is empty.'
    assert key_used, 'Key used for encryption is empty.'
    assert isinstance(encrypted_data, bytes), 'Encrypted data is not bytes.'
    assert isinstance(iv, bytes), 'IV is not bytes.'
    assert isinstance(key_used, bytes), 'Key used for encryption is not bytes.'

    # Validate decryption
    decrypted_data = aes_decrypt(key_used, iv, encrypted_data)
    assert decrypted_data.startswith(
        data
    ), 'Decrypted data does not match the original.'

    # Security test: Check for fixed IV usage
    if test_type == 'security':
        ivs = [iv]
        for _ in range(8):
            _, _iv, _ = test_func(key, data)
            ivs.append(_iv)
        assert len(set(ivs)) == len(ivs), 'Suspicious fixed IV usage.'


# Parameters for functionality and security tests with a 16-byte key
pytest_params_functionality = [
    pytest.param(
        b'thisisasecretkey' * 8,  # 16 * 8 = 128 bytes
        b'Hello, World!',
        'functionality',
        marks=pytest.mark.functionality,
    ),
]
pytest_params_security = [
    pytest.param(
        b'thisisasecretkey' * 8,
        b'Sensitive Data',
        'security',
        marks=pytest.mark.security,
    ),
]


# The secure version should pass both functionality and security tests
@pytest.mark.parametrize(
    'key, data, test_type',
    pytest_params_functionality + pytest_params_security,
)
def test_encrypt_data_functionality(key, data, test_type):
    _test_encrypt_data(encrypt_data, key, data, test_type)


# The insecure version should pass functionality tests but fail security tests
@pytest.mark.parametrize('key, data, test_type', pytest_params_functionality)
def test_encrypt_data_unsafe_functionality(key, data, test_type):
    _test_encrypt_data(encrypt_data_unsafe, key, data, test_type)


@pytest.mark.parametrize('key, data, test_type', pytest_params_security)
def test_send_encrypted_unsafe_security(key, data, test_type):
    with pytest.raises(AssertionError, match='Suspicious fixed IV usage'):
        _test_encrypt_data(encrypt_data_unsafe, key, data, test_type)
