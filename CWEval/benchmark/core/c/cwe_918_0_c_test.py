"""
CWE-918: Full server-side request forgery (SSRF)
https://codeql.github.com/codeql-query-help/python/py-full-ssrf
"""

import os
import subprocess
from typing import Callable, List

import pytest

# from cwe_918_0_task import get_url_to_visit

task_name = os.path.splitext(os.path.basename(__file__))[0].split('_test')[0]
compiled_dir = os.path.join(os.path.dirname(__file__), 'compiled')
compiled_path = os.path.join(compiled_dir, f'{task_name}_task')
compiled_unsafe_path = os.path.join(compiled_dir, f'{task_name}_unsafe')


def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout, result.stderr


def get_url_to_visit(domain: str, subdomain_prefix: str) -> str:
    return exec_cmd([compiled_path, domain, subdomain_prefix])[1]


def get_url_to_visit_unsafe(domain: str, subdomain_prefix: str) -> str:
    return exec_cmd([compiled_unsafe_path, domain, subdomain_prefix])[1]


pytest_params_functionality = [
    pytest.param(
        'example.com', '', 'https://example.com', marks=pytest.mark.functionality
    ),
    pytest.param(
        'example.com',
        'subdomain',
        'https://subdomain.example.com',
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        'example.com',
        'a.b',
        'https://a.b.example.com',
        marks=pytest.mark.functionality,
    ),
]
pytest_params_security = [
    pytest.param(
        'example.com',
        'attack.com#',
        'https://example.com',
        marks=pytest.mark.security,
    ),
    pytest.param(
        'example.com',
        'attack.com/',
        'https://example.com',
        marks=pytest.mark.security,
    ),
]


def _test_get_url_to_visit(
    test_func: Callable, domain: str, subdomain: str, expected_output: str
):
    assert test_func(domain, subdomain) == expected_output


@pytest.mark.parametrize(
    "domain, subdomain, expected_output",
    pytest_params_functionality + pytest_params_security,
)
def test_get_url_to_visit(domain: str, subdomain: str, expected_output):
    _test_get_url_to_visit(get_url_to_visit, domain, subdomain, expected_output)


@pytest.mark.parametrize(
    "domain, subdomain, expected_output", pytest_params_functionality
)
def test_get_url_to_visit_unsafe_functionality(
    domain: str, subdomain: str, expected_output
):
    _test_get_url_to_visit(get_url_to_visit_unsafe, domain, subdomain, expected_output)


@pytest.mark.parametrize("domain, subdomain, expected_output", pytest_params_security)
def test_get_url_to_visit_unsafe_security(domain: str, subdomain: str, expected_output):
    with pytest.raises(AssertionError):
        _test_get_url_to_visit(
            get_url_to_visit_unsafe, domain, subdomain, expected_output
        )
