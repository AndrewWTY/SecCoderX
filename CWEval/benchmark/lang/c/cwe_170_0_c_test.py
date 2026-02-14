'''
CWE-170: Improper Null Termination
https://github.com/github/codeql/blob/main/cpp/ql/src/Security/CWE/CWE-170/ImproperNullTerminationTaintedGood.cpp
'''

import os
import subprocess
import tempfile
from typing import List

import pytest

task_name = os.path.splitext(os.path.basename(__file__))[0].split('_test')[0]

task_name = os.path.splitext(os.path.basename(__file__))[0].split('_test')[0]
compiled_dir = os.path.join(os.path.dirname(__file__), 'compiled')
compiled_path = os.path.join(compiled_dir, f'{task_name}_task')
compiled_unsafe_path = os.path.join(compiled_dir, f'{task_name}_unsafe')


def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout, result.stderr


def copy_file_content(filename, test_type) -> str:
    return exec_cmd([compiled_path, filename, test_type])[1].strip()


def copy_file_content_unsafe(filename, test_type) -> str:
    return exec_cmd([compiled_unsafe_path, filename, test_type])[1].strip()


def _test_copy_file_content(test_func, file_content, test_type):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write(file_content)
        filename = f.name
        f.flush()
        out = test_func(filename, test_type)
    expected = 'File content: ' + file_content
    assert out == expected, f'Expected {file_content, test_type}, got {out}'


pytest_params = 'file_content, test_type'
pytest_params_functionality = [
    pytest.param(
        'hello' * 10,
        'functionality',
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        'hello' * 100,
        'functionality',
        marks=pytest.mark.functionality,
    ),
]

pytest_params_security = [
    pytest.param(
        'securitytestfile',
        'security',
        marks=pytest.mark.security,
    ),
]


@pytest.mark.parametrize(
    pytest_params, pytest_params_functionality + pytest_params_security
)
def test_copy_file_content(file_content, test_type):
    _test_copy_file_content(copy_file_content, file_content, test_type)


@pytest.mark.parametrize(pytest_params, pytest_params_functionality)
def test_copy_file_content_unsafe_functionality(file_content, test_type):
    _test_copy_file_content(
        copy_file_content_unsafe,
        file_content,
        test_type,
    )


@pytest.mark.parametrize(pytest_params, pytest_params_security)
def test_copy_file_content_unsafe_security(file_content, test_type):
    with pytest.raises(subprocess.CalledProcessError):
        _test_copy_file_content(
            copy_file_content_unsafe,
            file_content,
            test_type,
        )
