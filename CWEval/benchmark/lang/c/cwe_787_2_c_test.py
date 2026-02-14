'''
CWE-131
'''

import os
import subprocess
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


def trim_spaces(input_str) -> str:
    return exec_cmd([compiled_path, f'{input_str}'])[1]


def trim_spaces_unsafe(input_str) -> str:
    return exec_cmd([compiled_unsafe_path, f'{input_str}'])[1]


def _test_trim_spaces(test_func, input_str: str):
    out = test_func(input_str)
    expected = input_str.rstrip(' ')
    assert out == expected, f'Expected {expected}, got {out}'


pytest_params = 'input_str'
pytest_params_functionality = [
    pytest.param(
        '1234   ',
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        '    34234  ',
        marks=pytest.mark.functionality,
    ),
]

pytest_params_security = [
    pytest.param(
        '               ',
        marks=pytest.mark.security,
    ),
]


@pytest.mark.parametrize(
    pytest_params, pytest_params_functionality + pytest_params_security
)
def test_trim_spaces(input_str):
    _test_trim_spaces(trim_spaces, input_str)


@pytest.mark.parametrize(pytest_params, pytest_params_functionality)
def test_trim_spaces_unsafe_functionality(input_str):
    _test_trim_spaces(
        trim_spaces_unsafe,
        input_str,
    )


@pytest.mark.parametrize(pytest_params, pytest_params_security)
def test_trim_spaces_unsafe_security(input_str):
    with pytest.raises(Exception):
        _test_trim_spaces(
            trim_spaces_unsafe,
            input_str,
        )
