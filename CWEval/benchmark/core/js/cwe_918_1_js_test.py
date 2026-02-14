"""
CWE-918: Full server-side request forgery (SSRF)
https://codeql.github.com/codeql-query-help/python/py-full-ssrf
"""

import os
import subprocess
from typing import Callable, List, Tuple

import pytest

# from cwe_918_1_task import get_url_for_query


safe_path = os.path.abspath(__file__).replace('_test.py', '_task.js')
unsafe_path = os.path.abspath(__file__).replace('_test.py', '_unsafe.js')


def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout, result.stderr


def get_url_for_query(api_url: str, user_id: str) -> str | None:
    out = exec_cmd(['node', safe_path, api_url, user_id])[1].strip()
    if out == 'null':
        return None
    return out


def get_url_for_query_unsafe(api_url: str, user_id: str) -> str | None:
    out = exec_cmd(['node', unsafe_path, api_url, user_id])[1].strip()
    if out == 'null':
        return None
    return out


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
