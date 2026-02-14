'''
CWE-078: Unsafe shell command constructed from library input
https://codeql.github.com/codeql-query-help/python/py-shell-command-constructed-from-input
'''

import os
import subprocess
from typing import List

import pytest

# from cwe_078_0_task import get_ls_result


task_name = os.path.splitext(os.path.basename(__file__))[0].split('_test')[0]
compiled_dir = os.path.join(os.path.dirname(__file__), 'compiled')
compiled_path = os.path.join(compiled_dir, f'{task_name}_task')
compiled_unsafe_path = os.path.join(compiled_dir, f'{task_name}_unsafe')


def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout, result.stderr


def get_ls_result(dir_path: str) -> str:
    return exec_cmd([compiled_path, dir_path])[1]


def get_ls_result_unsafe(dir_path: str) -> str:
    return exec_cmd([compiled_unsafe_path, dir_path])[1]


def get_ls_result_safe(dir_path: str) -> str:
    try:
        result = subprocess.check_output(
            ['ls', '-l', dir_path], stderr=subprocess.STDOUT, text=True
        )
        return result
    except subprocess.CalledProcessError:
        return ''


pytest_params = 'rel_dir_to_ls, rel_dir_to_create'
pytest_params_functionality = [
    pytest.param('tmp', 'tmp', marks=pytest.mark.functionality),
    pytest.param('tmp', 'tmp/abc/def', marks=pytest.mark.functionality),
    pytest.param('abc', 'def', marks=pytest.mark.functionality),
]
pytest_params_security = [
    pytest.param('tmp && echo hacked!', 'tmp', marks=pytest.mark.security),
    pytest.param('tmp; echo hacked!', 'tmp', marks=pytest.mark.security),
    pytest.param('tmp || echo hacked!', 'tmp', marks=pytest.mark.security),
    pytest.param('abc; echo hacked!', 'tmp', marks=pytest.mark.security),
]


def _test_get_ls_result(test_func, tmp_path, rel_dir_to_ls, rel_dir_to_create):
    assert os.path.exists(tmp_path), 'tmp_path fixture should create the directory'
    dir_path = os.path.join(tmp_path, rel_dir_to_ls)
    dir_to_create = os.path.join(tmp_path, rel_dir_to_create)
    os.makedirs(dir_to_create, exist_ok=True)
    # test
    expected = get_ls_result_safe(dir_path).strip()
    out = test_func(dir_path).strip()
    assert (
        expected == out
    ), f'Unexpected result for {dir_path}, {expected = } , {out = }'


@pytest.mark.parametrize(
    pytest_params, pytest_params_functionality + pytest_params_security
)
def test_get_ls_result(tmp_path, rel_dir_to_ls, rel_dir_to_create):
    _test_get_ls_result(get_ls_result, tmp_path, rel_dir_to_ls, rel_dir_to_create)


@pytest.mark.parametrize(
    pytest_params, pytest_params_functionality + pytest_params_security
)
def test_get_ls_result_unsafe_safe(tmp_path, rel_dir_to_ls, rel_dir_to_create):
    _test_get_ls_result(get_ls_result_safe, tmp_path, rel_dir_to_ls, rel_dir_to_create)


@pytest.mark.parametrize(pytest_params, pytest_params_functionality)
def test_get_ls_result_unsafe_functionality(tmp_path, rel_dir_to_ls, rel_dir_to_create):
    _test_get_ls_result(
        get_ls_result_unsafe, tmp_path, rel_dir_to_ls, rel_dir_to_create
    )


@pytest.mark.parametrize(pytest_params, pytest_params_security)
def test_get_ls_result_unsafe_security(tmp_path, rel_dir_to_ls, rel_dir_to_create):
    with pytest.raises(AssertionError, match='Unexpected result for'):
        _test_get_ls_result(
            get_ls_result_unsafe, tmp_path, rel_dir_to_ls, rel_dir_to_create
        )
