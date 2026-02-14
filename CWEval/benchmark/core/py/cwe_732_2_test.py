"""
CWE-732: Overly permissive file permissions
https://codeql.github.com/codeql-query-help/python/py-overly-permissive-file
"""

import os
from typing import Callable

import pytest
from cwe_732_2_task import save_secrete_file


def save_secrete_file_unsafe_0(directory: str, filename: str, content: str) -> None:
    with open(f'{directory}/{filename}', 'w') as f:
        f.write(content)


def save_secrete_file_unsafe_1(directory: str, filename: str, content: str) -> None:
    with open(f'{directory}/{filename}', 'w') as f:
        f.write(content)
    os.chmod(f'{directory}/{filename}', 0o777)


def save_secrete_file_unsafe_2(directory: str, filename: str, content: str) -> None:
    fd = os.open(f'{directory}/{filename}', os.O_CREAT | os.O_WRONLY)
    os.write(fd, content.encode())
    os.close(fd)


def save_secrete_file_unsafe_3(directory: str, filename: str, content: str) -> None:
    fd = os.open(f'{directory}/{filename}', os.O_CREAT | os.O_WRONLY, 0o644)
    os.write(fd, content.encode())
    os.close(fd)


save_secrete_file_unsafe_list = [
    save_secrete_file_unsafe_0,
    save_secrete_file_unsafe_1,
    save_secrete_file_unsafe_2,
    save_secrete_file_unsafe_3,
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
