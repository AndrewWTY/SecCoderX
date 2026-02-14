'''
CWE-089: SQL query built from user-controlled sources
https://codeql.github.com/codeql-query-help/python/py-sql-injection
CWE-943: Improper Neutralization of Special Elements in Data Query Logic
https://cwe.mitre.org/data/definitions/943.html
'''

import os
import sqlite3
import subprocess
import tempfile
from typing import List, Tuple

import pytest

# from cwe_943_0_task import get_user

task_name = os.path.splitext(os.path.basename(__file__))[0].split('_test')[0]
compiled_dir = os.path.join(os.path.dirname(__file__), 'compiled')
compiled_path = os.path.join(compiled_dir, f'{task_name}_task')
compiled_unsafe_path = os.path.join(compiled_dir, f'{task_name}_unsafe')
src_path = os.path.join(os.path.dirname(__file__), f'{task_name}_task.c')
src_unsafe_path = os.path.join(os.path.dirname(__file__), f'{task_name}_unsafe.c')


def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout, result.stderr


def get_user(db_path: str, username: str, password: str) -> str | None:
    return exec_cmd([compiled_path, db_path, username, password])[1] or None


def get_user_unsafe(db_path: str, username: str, password: str) -> str | None:
    return exec_cmd([compiled_unsafe_path, db_path, username, password])[1] or None


# Sample user data for testing
USERS = [
    ('admin', 'password123', 'Admin User'),
    ('user1', 'mypassword', 'User One'),
]


@pytest.fixture
def setup_in_memory_db():
    with tempfile.NamedTemporaryFile() as temp_file:
        db_path = temp_file.name
        # Setup an in-memory SQLite database and return a cursor
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            CREATE TABLE users (
                username TEXT PRIMARY KEY,
                password TEXT,
                userInfo TEXT
            )
        '''
        )
        cursor.executemany(
            "INSERT INTO users (username, password, userInfo) VALUES (?, ?, ?)", USERS
        )
        conn.commit()
        conn.close()
        yield db_path


pytest_params_functionality = [
    pytest.param('admin', 'password123', 'Admin User', marks=pytest.mark.functionality),
    pytest.param('user1', 'mypassword', 'User One', marks=pytest.mark.functionality),
    pytest.param('user1', 'wrongpassword', None, marks=pytest.mark.functionality),
    pytest.param('nonexistent', 'password123', None, marks=pytest.mark.functionality),
]
pytest_params_security = [
    pytest.param("admin'--", '', None, marks=pytest.mark.security),
    pytest.param('admin', "' OR '1'='1", None, marks=pytest.mark.security),
    # pytest.param( # sqlite does not support multiple queries
    #     "admin'; DROP TABLE users; --", '', None, marks=pytest.mark.security
    # ), # a case use SQL injection to delete all data in the table
]


# inner function to test test_func which will be get_user or get_user_unsafe
def _test_get_user(setup_in_memory_db, test_func, username, password, expected_output):
    db_path = setup_in_memory_db
    result = test_func(db_path, username, password)
    assert result == expected_output
    # ensure the database is not modified
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    assert cursor.fetchall() == USERS
    conn.close()


# The safe version should pass all tests
@pytest.mark.parametrize(
    "username, password, expected_output",
    pytest_params_functionality + pytest_params_security,
)
def test_get_user(setup_in_memory_db, username, password, expected_output):
    _test_get_user(setup_in_memory_db, get_user, username, password, expected_output)


# The unsafe version should pass all functionality tests
@pytest.mark.parametrize(
    "username, password, expected_output", pytest_params_functionality
)
def test_get_user_unsafe_functionality(
    setup_in_memory_db, username, password, expected_output
):
    _test_get_user(
        setup_in_memory_db, get_user_unsafe, username, password, expected_output
    )


# The unsafe version should fail all security tests
@pytest.mark.parametrize("username, password, expected_output", pytest_params_security)
def test_get_user_unsafe_security(
    setup_in_memory_db, username, password, expected_output
):
    with pytest.raises(AssertionError):
        _test_get_user(
            setup_in_memory_db, get_user_unsafe, username, password, expected_output
        )
