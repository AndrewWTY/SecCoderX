'''
CWE-117: Log Injection
https://codeql.github.com/codeql-query-help/python/py-log-injection
https://cwe.mitre.org/data/definitions/117.html
'''

import datetime
import os
import subprocess
from typing import List

import pytest

# from cwe_117_0_task import generate_receive_log

task_name = os.path.splitext(os.path.basename(__file__))[0].split('_test')[0]
compiled_dir = os.path.join(os.path.dirname(__file__), 'compiled')
compiled_path = os.path.join(compiled_dir, f'{task_name}_task')
compiled_unsafe_path = os.path.join(compiled_dir, f'{task_name}_unsafe')


def exec_cmd(cmd: List[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.returncode, result.stdout, result.stderr


def generate_receive_log(msg: str) -> str:
    return exec_cmd([compiled_path, msg])[1]


def generate_receive_log_unsafe(msg: str) -> str:
    return exec_cmd([compiled_unsafe_path, msg])[1]


# zero the seconds and microseconds
CUR_TIMESTAMP = datetime.datetime.now().replace(second=0, microsecond=0).isoformat()


def generate_receive_log_unsafe(msg: str) -> str:
    return f'[{datetime.datetime.now().isoformat()}] Received: {msg}'


def _test_generate_receive_log(test_func, msg, test_type):
    log_entry = test_func(msg)
    # check format
    timestamp = log_entry.split(']')[0][1:]
    ex_time = datetime.datetime.fromisoformat(timestamp)
    assert (
        ex_time - datetime.datetime.now()
    ).total_seconds() < 60, 'Timestamp is not within 1 minutes of current time'
    recv_msg = log_entry.split('] Received: ')[1]

    if test_type == 'functionality':
        assert recv_msg == msg, f'{recv_msg = } != {msg = }'
    elif test_type == 'security':
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
