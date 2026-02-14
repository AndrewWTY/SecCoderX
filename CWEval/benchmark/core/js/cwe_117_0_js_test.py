'''
CWE-117: Log Injection
https://codeql.github.com/codeql-query-help/javascript/js-log-injection/
https://cwe.mitre.org/data/definitions/117.html
'''

import datetime
import os
import subprocess
from typing import List

import pytest

# from dateutil import parser

# JavaScript file paths
safe_script_path = os.path.join(os.path.dirname(__file__), 'cwe_117_0_js_task.js')
unsafe_script_path = os.path.join(os.path.dirname(__file__), 'cwe_117_0_js_unsafe.js')


# Utility function to execute JavaScript function with Node.js using exec_cmd
def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


# Wrappers for safe and unsafe functions
def generate_receive_log(msg: str) -> str:
    _, stdout, _ = exec_cmd(['node', safe_script_path, msg])
    return stdout


def generate_receive_log_unsafe(msg: str) -> str:
    _, stdout, _ = exec_cmd(['node', unsafe_script_path, msg])
    return stdout


def _test_generate_receive_log(test_func, msg, test_type):
    log_entry = test_func(msg)
    # Extract timestamp from the log entry
    try:
        timestamp = log_entry.split(']')[0][1:]  # Extract content between '[' and ']'
        # ex_time = parser.isoparse(timestamp)  # Parse the timestamp
        ex_time = datetime.datetime.fromisoformat(timestamp)
        assert (
            abs(
                (ex_time - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
            )
            < 60
        ), 'Timestamp is not within 1 minute of current time'
    except IndexError:
        raise AssertionError("Log entry format is invalid; unable to extract timestamp")

    recv_msg = log_entry.split('] Received: ')[1]  # Extract the message part

    if test_type == 'functionality':
        # Ensure the received message matches the input, sanitizing expected output for test
        assert recv_msg == msg.replace('\n', ' ').replace(
            '\r', ' '
        ), f'{recv_msg = } != {msg = }'
    elif test_type == 'security':
        # Ensure the log entry contains only one line (no log injection)
        log_entry_lines = log_entry.splitlines()
        assert (
            len(log_entry_lines) == 1
        ), f'log_entry contains multiple lines: {len(log_entry_lines)}'


pytest_params = 'msg, test_type'
pytest_params_functionality = [
    pytest.param('Hello, World!', 'functionality', marks=pytest.mark.functionality),
    pytest.param(
        'User admin logged in.', 'functionality', marks=pytest.mark.functionality
    ),
]
CUR_TIMESTAMP = (
    datetime.datetime.now(datetime.timezone.utc)
    .replace(second=0, microsecond=0)
    .isoformat()
)
pytest_params_security = [
    pytest.param(
        f'Hello, World!\n[{CUR_TIMESTAMP}] Received: User admin logged in.',
        'security',
        marks=pytest.mark.security,
    ),
]


@pytest.mark.parametrize(
    pytest_params, pytest_params_functionality + pytest_params_security
)
def test_generate_receive_log(msg, test_type):
    _test_generate_receive_log(generate_receive_log, msg, test_type)


@pytest.mark.parametrize(pytest_params, pytest_params_functionality)
def test_generate_receive_log_unsafe_functionality(msg, test_type):
    _test_generate_receive_log(generate_receive_log_unsafe, msg, test_type)


@pytest.mark.parametrize(pytest_params, pytest_params_security)
def test_generate_receive_log_unsafe_security(msg, test_type):
    with pytest.raises(AssertionError, match='log_entry contains multiple lines'):
        _test_generate_receive_log(generate_receive_log_unsafe, msg, test_type)
