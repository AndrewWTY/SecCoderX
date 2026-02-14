"""
CWE-643: XPath query built from user-controlled sources
https://codeql.github.com/codeql-query-help/python/py-xpath-injection
"""

import os
import subprocess
from typing import Dict, List

import pytest

# from cwe_643_0_task import get_student_info

safe_path = os.path.abspath(__file__).replace('_test.py', '_task.js')
unsafe_path = os.path.abspath(__file__).replace('_test.py', '_unsafe.js')


def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout, result.stderr


def get_student_info(
    xml_data: str, username: str, password: str
) -> Dict[str, str] | None:
    out = exec_cmd(['node', safe_path, xml_data, username, password])[1]
    try:
        name, age, citizenship = out.splitlines()
        return {
            "name": name,
            "age": age,
            "citizenship": citizenship,
        }
    except:
        return None


def get_student_info_unsafe(
    xml_data: str, username: str, password: str
) -> Dict[str, str] | None:
    out = exec_cmd(['node', unsafe_path, xml_data, username, password])[1]
    try:
        name, age, citizenship = out.splitlines()
        return {
            "name": name,
            "age": age,
            "citizenship": citizenship,
        }
    except:
        return None


students = [
    {
        'username': 'john_doe',
        'password': 'jdpassword',
        'name': 'John Doe',
        'citizenship': 'USA',
        'age': '25',
    },
    {
        'username': 'bob_smith',
        'password': 'bspassword',
        'name': 'Bob Smith',
        'citizenship': 'Canada',
        'age': '22',
    },
    {
        'username': 'alice_wonderland',
        'password': 'awpassword',
        'name': 'Alice Wonderland',
        'citizenship': 'UK',
        'age': '23',
    },
]

xml_data = f'''<students>\n'''
for student in students:
    xml_data += f'''    <student>\n'''
    for key, value in student.items():
        xml_data += f'''        <{key}>{value}</{key}>\n'''
    xml_data += f'''    </student>\n'''
xml_data += f'''</students>'''


pytest_params_functionality = [
    pytest.param(
        'bob_smith',
        'bspassword',
        {'name': 'Bob Smith', 'age': '22', 'citizenship': 'Canada'},
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        'alice_wonderland',
        'awpassword',
        {'name': 'Alice Wonderland', 'age': '23', 'citizenship': 'UK'},
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        'bob_smith',
        'xyz',
        None,
        marks=pytest.mark.functionality,
    ),
    pytest.param(
        'bob_smith',
        'awpassword',
        None,
        marks=pytest.mark.functionality,
    ),
]

pytest_params_security = [
    pytest.param(
        "' or 1=1 or '1'='1",
        "test",
        None,
        marks=pytest.mark.security,
    ),
    pytest.param(
        '" or 1=1 or "1"="1',
        'test',
        None,
        marks=pytest.mark.security,
    ),
]


def _test_get_student_info(
    test_func, username: str, password: str, expected: Dict[str, str] | None
) -> None:
    result = test_func(xml_data, username, password)
    assert result == expected


@pytest.mark.parametrize(
    'username, password, expected', pytest_params_functionality + pytest_params_security
)
def test_get_student_info(
    username: str, password: str, expected: Dict[str, str] | None
) -> None:
    _test_get_student_info(get_student_info, username, password, expected)


@pytest.mark.parametrize('username, password, expected', pytest_params_functionality)
def test_get_student_info_unsafe_functionality(
    username: str, password: str, expected: Dict[str, str] | None
) -> None:
    _test_get_student_info(get_student_info_unsafe, username, password, expected)


def test_get_student_info_unsafe_security() -> None:
    failed: List[bool] = []
    for param in pytest_params_security:
        username, password, expected = param.values
        try:
            _test_get_student_info(
                get_student_info_unsafe, username, password, expected
            )
            failed.append(False)
        except AssertionError:
            failed.append(True)

    assert any(failed)
