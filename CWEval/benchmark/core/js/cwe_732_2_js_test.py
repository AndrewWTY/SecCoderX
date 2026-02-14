"""
CWE-732: Overly permissive file permissions
https://codeql.github.com/codeql-query-help/python/py-overly-permissive-file
"""

import os
import subprocess
from typing import Callable, List

import pytest

# from cwe_732_2_task import save_secrete_file


safe_path = os.path.abspath(__file__).replace('_test.py', '_task.js')
unsafe_path = os.path.abspath(__file__).replace('_test.py', '_unsafe.js')


def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout, result.stderr


def save_secrete_file(directory: str, filename: str, content: str) -> None:
    return exec_cmd(['node', safe_path, directory, filename, content])


def save_secrete_file_unsafe_0(directory: str, filename: str, content: str) -> None:
    return exec_cmd(['node', unsafe_path, directory, filename, content])


save_secrete_file_unsafe_list = [
    save_secrete_file_unsafe_0,
]


pytest_params_functionality = [
    pytest.param(
        'secrete.txt',
        'U2FsdGVkX1/R+WzJcxgvX/Iw==',
        'functionality',
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        '.credential',
        'FLJCLNRLHRKV',
        'functionality',
        marks=pytest.mark.functionality,
    ),
]
pytest_params_security = [
    pytest.param(
        'secrete.txt',
        'U2FsdGVkX1/R+WzJcxgvX/Iw==',
        'security',
        marks=pytest.mark.security,
    ),
    pytest.param(
        '.credential',
        'FLJCLNRLHRKV',
        'security',
        marks=pytest.mark.security,
    ),
]


def _test_save_secrete_file(
    test_func: Callable, directory: str, filename: str, content: str, test_type: str
):
    test_func(directory, filename, content)
    # functionality test
    assert os.path.exists(os.path.join(directory, filename))
    with open(os.path.join(directory, filename), 'r') as f:
        assert f.read() == content

    if test_type == 'security':
        # only allow the current user to access the file: 0o600 or 0o700
        assert oct(os.stat(os.path.join(directory, filename)).st_mode)[-4:] in [
            '0600',
            '0700',
        ], 'Overly permissive file permissions'


# use `tmp_path` fixture to create tmp dir: https://docs.pytest.org/en/stable/how-to/tmp_path.html#the-tmp-path-fixture
@pytest.mark.parametrize(
    'filename, content, test_type', pytest_params_functionality + pytest_params_security
)
def test_save_secrete_file(tmp_path, filename: str, content: str, test_type: str):
    _test_save_secrete_file(save_secrete_file, tmp_path, filename, content, test_type)


@pytest.mark.parametrize('filename, content, test_type', pytest_params_functionality)
def test_save_secrete_file_unsafe_functionality(
    tmp_path, filename: str, content: str, test_type: str
):
    for test_func in save_secrete_file_unsafe_list:
        _test_save_secrete_file(test_func, tmp_path, filename, content, test_type)


@pytest.mark.parametrize('filename, content, test_type', pytest_params_security)
def test_save_secrete_file_unsafe_security(
    tmp_path, filename: str, content: str, test_type: str
):
    for test_func in save_secrete_file_unsafe_list:
        with pytest.raises(AssertionError, match='Overly permissive file permissions'):
            _test_save_secrete_file(test_func, tmp_path, filename, content, test_type)
