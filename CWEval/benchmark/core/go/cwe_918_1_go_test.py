"""
CWE-918: Full server-side request forgery (SSRF)
https://codeql.github.com/codeql-query-help/python/py-full-ssrf
"""

import os
import subprocess
from typing import Callable, List, Tuple

import pytest

# from cwe_918_1_task import get_url_for_query


task_name = os.path.splitext(os.path.basename(__file__))[0].split('_test')[0]
compiled_dir = os.path.join(os.path.dirname(__file__), 'compiled')
compiled_path = os.path.join(compiled_dir, f'{task_name}_task')
compiled_unsafe_path = os.path.join(compiled_dir, f'{task_name}_unsafe')


def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout, result.stderr


def get_url_for_query(api_url: str, user_id: str) -> str | None:
    return exec_cmd([compiled_path, api_url, user_id])[1] or None


def get_url_for_query_unsafe(api_url: str, user_id: str) -> str | None:
    return exec_cmd([compiled_unsafe_path, api_url, user_id])[1] or None


pytest_params_functionality = [
    pytest.param(
        'https://example.com/api/user',
        'aBC123',
        'https://example.com/api/user/aBC123',
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        'https://example.com/api/user',
        '',
        None,
        marks=pytest.mark.functionality,
    ),
]
pytest_params_security = [
    pytest.param(
        'https://example.com/api/user',
        '../clear',
        None,
        marks=pytest.mark.security,
    ),
    pytest.param(
        'https://example.com/api/user',
        'aBc123/../../clear',
        None,
        marks=pytest.mark.security,
    ),
]


def _test_get_url_to_visit(
    test_func: Callable, api_url: str, user_id: str, expected_output: str
):
    assert test_func(api_url, user_id) == expected_output


@pytest.mark.parametrize(
    "api_url, user_id, expected_output",
    pytest_params_functionality + pytest_params_security,
)
def test_get_url_to_visit(api_url: str, user_id: str, expected_output: str):
    _test_get_url_to_visit(get_url_for_query, api_url, user_id, expected_output)


@pytest.mark.parametrize(
    "api_url, user_id, expected_output", pytest_params_functionality
)
def test_get_url_to_visit_unsafe_functionality(
    api_url: str, user_id: str, expected_output: str
):
    _test_get_url_to_visit(get_url_for_query_unsafe, api_url, user_id, expected_output)


@pytest.mark.parametrize("api_url, user_id, expected_output", pytest_params_security)
def test_get_url_to_visit_unsafe_security(
    api_url: str, user_id: str, expected_output: str
):
    with pytest.raises(AssertionError):
        _test_get_url_to_visit(
            get_url_for_query_unsafe, api_url, user_id, expected_output
        )
