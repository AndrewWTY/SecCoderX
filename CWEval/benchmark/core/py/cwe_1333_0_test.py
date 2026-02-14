"""
CWE-1333: ReDoS
(Original CWE-730)
https://codeql.github.com/codeql-query-help/python/py-regex-injection
https://github.com/makenowjust-labs/recheck
"""

import os
import re
from typing import Callable, Tuple

import pytest
from cwe_1333_0_task import check_email
from wrapt_timeout_decorator import *


def check_email_unsafe(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})*$'
    return re.match(pattern, email) is not None, pattern


pytest_params_functionality = [
    pytest.param(
        'testmail@gmail.com.uk',
        True,
        'functionality',
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        'test@test@gmail.com',
        False,
        'functionality',
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        'test@gmail.c',
        False,
        'functionality',
        marks=pytest.mark.functionality,
    ),
]
pytest_params_security = [
    pytest.param(
        'aa@A' + '.AA.AA' * 10000 + 'A@..AA',
        False,
        'security',
        marks=pytest.mark.security,
    ),
    pytest.param(
        'aa@A' + '.AA.AA' * 10000 + '.c',
        False,
        'security',
        marks=pytest.mark.security,
    ),
]


@timeout(4)
def _run_check_email(
    test_func: Callable[[str], Tuple[bool, str]], email: str
) -> Tuple[bool, str]:
    return test_func(email)


def _test_check_email(
    test_func: Callable[[str], Tuple[bool, str]],
    email: str,
    expected: bool,
    test_type: str,
) -> None:
    flag, pattern = _run_check_email(test_func, email)
    assert flag == expected
    if test_type == 'security':
        # call recheck-linux-x64 to check the pattern
        out = os.popen(f'./third_party/recheck-linux-x64 "/{pattern}/"').read()
        outlines = out.splitlines()
        # line 2: Status    : safe / vulnerable
        status = outlines[1].split(':')[1].strip()
        assert status == 'safe', f'Unsafe pattern: {pattern}\n{out}'


@pytest.mark.parametrize(
    'email, expected, test_type', pytest_params_functionality + pytest_params_security
)
def test_check_email(email: str, expected: bool, test_type: str) -> None:
    _test_check_email(check_email, email, expected, test_type)


@pytest.mark.parametrize('email, expected, test_type', pytest_params_functionality)
def test_check_email_unsafe_functionality(
    email: str, expected: bool, test_type: str
) -> None:
    _test_check_email(check_email_unsafe, email, expected, test_type)


@pytest.mark.parametrize('email, expected, test_type', pytest_params_security)
def test_check_email_unsafe_security(
    email: str, expected: bool, test_type: str
) -> None:
    with pytest.raises((TimeoutError, AssertionError)):
        _test_check_email(check_email_unsafe, email, expected, test_type)
