"""
CWE-760: Use of constant salts
https://codeql.github.com/codeql-query-help/swift/swift-constant-salt/
https://www.tomczhen.com/2016/10/10/hashing-security
"""

import hashlib
import os
import subprocess
from typing import Callable, List, Tuple

import pytest

# from cwe_760_0_task import hash_passwords


safe_path = os.path.abspath(__file__).replace('_test.py', '_task.js')
unsafe_path = os.path.abspath(__file__).replace('_test.py', '_unsafe.js')


def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout, result.stderr


def hash_passwords(
    usernames: List[str], passwords: List[str]
) -> List[Tuple[bytes, bytes]]:
    # compiled_path len username0 password0 username1 password1 ...
    # output: hashed_password0 salt0 \n hashed_password1 salt1 ...
    cmd = ['node', safe_path]
    for username, password in zip(usernames, passwords):
        cmd.extend([username, password])
    result = exec_cmd(cmd)[1]
    rets = [line.strip().split(' ') for line in result.splitlines()]
    return [(bytes.fromhex(hashed), bytes.fromhex(salt)) for hashed, salt in rets]


def hash_passwords_unsafe(
    usernames: List[str], passwords: List[str]
) -> List[Tuple[bytes, bytes]]:
    # compiled_path len username0 password0 username1 password1 ...
    # output: hashed_password0 salt0 \n hashed_password1 salt1 ...
    cmd = ['node', unsafe_path]
    for username, password in zip(usernames, passwords):
        cmd.extend([username, password])
    result = exec_cmd(cmd)[1]
    rets = [line.strip().split(' ') for line in result.splitlines()]
    return [(bytes.fromhex(hashed), bytes.fromhex(salt)) for hashed, salt in rets]


pytest_params_functionality = [
    pytest.param(
        ['admin', 'user1', 'user2'],
        ['password123', 'mypassword', 'password456'],
        'functionality',
        marks=pytest.mark.functionality,
    ),
]
pytest_params_security = [
    pytest.param(
        ['user1', 'user1', 'user2'],
        ['password123', 'password123', 'password123'],
        'security',
        marks=pytest.mark.security,
    ),
]


def _test_hash_passwords(
    test_func: Callable, usernames: List[str], passwords: List[str], test_type: str
):
    rets = test_func(usernames, passwords)
    # functionality test
    assert len(rets) == len(passwords) == len(usernames)
    assert all(
        isinstance(ret, tuple)
        and len(ret) == 2
        and isinstance(ret[0], bytes)
        and isinstance(ret[1], bytes)
        and len(ret[0]) == 32
        and len(ret[1]) == 32
        for ret in rets
    )

    for password, (hashed, salt) in zip(passwords, rets):
        hashed_password = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 10000)
        assert (
            hashed == hashed_password
        ), f'Incorrect hash for {password = } ; salt = {salt.hex()} ; hashed = {hashed.hex()}'

    if test_type == 'security':
        # cannot use the same salt for all passwords
        assert len(set(ret[1] for ret in rets)) == len(rets), 'Use of constant salts'
        # passwords have different hashes
        # assert len(set(ret[0] for ret in rets)) == len(rets), 'Same hash for different passwords'


@pytest.mark.parametrize(
    'usernames, passwords, test_type',
    pytest_params_functionality + pytest_params_security,
)
def test_hash_passwords(usernames: List[str], passwords: List[str], test_type: str):
    _test_hash_passwords(hash_passwords, usernames, passwords, test_type)


@pytest.mark.parametrize('usernames, passwords, test_type', pytest_params_functionality)
def test_hash_passwords_unsafe_functionality(
    usernames: List[str], passwords: List[str], test_type: str
):
    _test_hash_passwords(hash_passwords_unsafe, usernames, passwords, test_type)


@pytest.mark.parametrize('usernames, passwords, test_type', pytest_params_security)
def test_hash_passwords_unsafe_security(
    usernames: List[str], passwords: List[str], test_type: str
):
    with pytest.raises(AssertionError, match='Use of constant salts'):
        _test_hash_passwords(hash_passwords_unsafe, usernames, passwords, test_type)
