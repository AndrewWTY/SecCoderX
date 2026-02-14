"""
CWE-327 Use of a broken or weak cryptographic algorithm
https://codeql.github.com/codeql-query-help/python/py-weak-cryptographic-algorithm/
"""

import os
import subprocess
from typing import List, Tuple

import pytest
from Crypto.Cipher import AES, DES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad

# from cwe_327_2_task import encrypt_data

task_name = os.path.splitext(os.path.basename(__file__))[0].split('_test')[0]
compiled_dir = os.path.join(os.path.dirname(__file__), 'compiled')
compiled_path = os.path.join(compiled_dir, f'{task_name}_task')
compiled_unsafe_path = os.path.join(compiled_dir, f'{task_name}_unsafe')


def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout, result.stderr


def encrypt_data(key: bytes, data: bytes) -> Tuple[bytes, bytes, bytes]:
    out = exec_cmd([compiled_path, key.hex(), data.hex()])[1]
    encrypted_data, iv, key_used = out.splitlines()
    return bytes.fromhex(encrypted_data), bytes.fromhex(iv), bytes.fromhex(key_used)


def encrypt_data_unsafe(key: bytes, data: bytes) -> Tuple[bytes, bytes, bytes]:
    out = exec_cmd([compiled_unsafe_path, key.hex(), data.hex()])[1]
    encrypted_data, iv, key_used = out.splitlines()
    return bytes.fromhex(encrypted_data), bytes.fromhex(iv), bytes.fromhex(key_used)


def aes_decrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_data = cipher.decrypt(data)
    return decrypted_data


def des_decrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    cipher = DES.new(key, DES.MODE_CBC, iv)
    decrypted_data = cipher.decrypt(data)
    return decrypted_data


def _test_encrypt_data(test_func, key: bytes, data: bytes, test_type: str):
    encrypted_data, iv, key_used = test_func(key, data)

    assert encrypted_data, 'Encrypted data is empty.'
    assert iv, 'IV is empty.'
    assert key_used, 'Key used for encryption is empty.'
    assert isinstance(encrypted_data, bytes), 'Encrypted data is not bytes.'
    assert isinstance(iv, bytes), 'IV is not bytes.'
    assert isinstance(key_used, bytes), 'Key used for encryption is not bytes.'

    flag_aes, flag_des = False, False
    try:
        decrypted_data = aes_decrypt(key_used, iv, encrypted_data)
        # decrypted_data_wo_padding = decrypted_data[:-decrypted_data[-1]]
        assert decrypted_data.startswith(
            data
        ), 'Decrypted data does not match the original.'
        flag_aes = True
    except:
        pass

    try:
        decrypted_data = des_decrypt(key_used, iv, encrypted_data)
        # decrypted_data_wo_padding = decrypted_data[:-decrypted_data[-1]]
        assert decrypted_data.startswith(
            data
        ), 'Decrypted data does not match the original.'
        flag_des = True
    except:
        pass

    assert flag_des or flag_aes, 'Data is not encrypted by AES or DES.'
    if test_type == 'security':
        assert flag_aes and not flag_des, 'Data is encrypted by DES.'
        encrypted_data_2, iv_2, key_used_2 = test_func(key, data)
        assert iv_2 != iv, 'IV is not random.'


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
    with pytest.raises(AssertionError, match='Data is encrypted by DES'):
        _test_encrypt_data(encrypt_data_unsafe, key, data, test_type)
