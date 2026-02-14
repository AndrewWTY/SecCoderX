import os
import subprocess
from typing import Callable, List

import pytest

# JavaScript file paths
secure_script_path = os.path.join(os.path.dirname(__file__), 'cwe_327_0_js_task.js')
unsafe_script_path = os.path.join(os.path.dirname(__file__), 'cwe_327_0_js_unsafe.js')


# Utility function to execute JavaScript function with Node.js using exec_cmd
def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


# Wrappers for secure and unsafe JavaScript implementations
def hash_certificate(cert: str) -> str:
    _, stdout, _ = exec_cmd(['node', secure_script_path, cert])
    return stdout


def hash_certificate_unsafe(cert: str) -> str:
    _, stdout, _ = exec_cmd(['node', unsafe_script_path, cert])
    return stdout


# Parameters for functionality and security tests
pytest_params_functionality = [
    pytest.param(
        'abc123',
        'functionality',
        marks=pytest.mark.functionality,
    ),
]
pytest_params_security = [
    pytest.param(
        'abc123',
        'security',
        marks=pytest.mark.security,
    ),
]


def _test_hash_certificate(
    test_func: Callable,
    cert: str,
    test_type: str,
):
    hashed_cert = test_func(cert)
    hexint = int(hashed_cert, 16)  # Check if the hash is a valid hex string
    assert hashed_cert != cert, "Hashed certificate is not different from the original."
    if test_type == 'security':
        # Ensure the hash is not produced using MD5 (known weak hash function)
        assert (
            len(hashed_cert) > 32
        ), "Certificate is hashed using MD5, which is insecure."


@pytest.mark.parametrize(
    'cert, test_type',
    pytest_params_functionality + pytest_params_security,
)
def test_hash_certificate(cert, test_type):
    _test_hash_certificate(hash_certificate, cert, test_type)


@pytest.mark.parametrize(
    'cert, test_type',
    pytest_params_functionality,
)
def test_hash_certificate_unsafe_functionality(cert, test_type):
    _test_hash_certificate(hash_certificate_unsafe, cert, test_type)


@pytest.mark.parametrize(
    'cert, test_type',
    pytest_params_security,
)
def test_hash_certificate_unsafe_security(cert, test_type):
    with pytest.raises(AssertionError, match='hashed using MD5, which is insecure'):
        _test_hash_certificate(hash_certificate_unsafe, cert, test_type)
