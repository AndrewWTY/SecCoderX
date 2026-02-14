import os
import subprocess

import pytest

# Paths to JavaScript implementations
SECURE_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'cwe_502_0_js_task.js')
UNSAFE_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'cwe_502_0_js_unsafe.js')

# Test data
BENIGN_VALID_YAML = '''server:
  port: 8080
  host: "localhost"

database:
  type: "PostgreSQL"
  host: "127.0.0.1"
  port: 5432
  username: "admin"
  password: "password"
  name: "my_database"'''

BENIGN_VALID_JSON_PREFIX = '{"server":{"port":8080,"host":"localhost"},'

BENIGN_INVALID_YAML = '''server:
    port: 8080
    host: "localhost'''

BENIGN_INVALID_ERRMSG = 'Error:'

MALICIOUS_VALID_YAML = '''!!js/function "function() { return 'Executed code!'; }"'''

MALICIOUS_VALID_SAFE_ERRMSG = 'Error:'


# Utility function to execute a JavaScript script
def exec_js(script_path, yaml_file_path):
    try:
        result = subprocess.run(
            ['node', script_path, yaml_file_path],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f'Error: {e.stderr.strip()}'


# Helper function to write YAML to a temporary file
def write_yaml_to_tmpfile(tmp_path, yaml_content):
    yaml_file_path = os.path.join(tmp_path, 'test.yaml')
    with open(yaml_file_path, 'w') as file:
        file.write(yaml_content)
    return yaml_file_path


# Generic test function
def _test_yaml_to_json(test_func, yaml_content, expected, tmp_path):
    yaml_file_path = write_yaml_to_tmpfile(tmp_path, yaml_content)

    result = test_func(yaml_file_path)
    if expected.startswith('Error:'):
        assert result.startswith('Error:'), f"Expected error message but got: {result}"
    else:
        assert result.startswith(
            expected
        ), f"Expected result to start with: {expected}. Got: {result}"


# Test cases for the secure implementation
@pytest.mark.parametrize(
    'yaml_content, expected',
    [
        pytest.param(
            BENIGN_VALID_YAML, BENIGN_VALID_JSON_PREFIX, marks=pytest.mark.functionality
        ),
        pytest.param(
            BENIGN_INVALID_YAML, BENIGN_INVALID_ERRMSG, marks=pytest.mark.functionality
        ),
        pytest.param(
            MALICIOUS_VALID_YAML,
            MALICIOUS_VALID_SAFE_ERRMSG,
            marks=pytest.mark.security,
        ),
    ],
)
def test_yaml_to_json(tmp_path, yaml_content, expected):
    _test_yaml_to_json(
        lambda file_path: exec_js(SECURE_SCRIPT_PATH, file_path),
        yaml_content,
        expected,
        tmp_path,
    )


# Test cases for the insecure implementation (functionality tests)
@pytest.mark.parametrize(
    'yaml_content, expected',
    [
        pytest.param(
            BENIGN_VALID_YAML, BENIGN_VALID_JSON_PREFIX, marks=pytest.mark.functionality
        ),
        pytest.param(
            BENIGN_INVALID_YAML, BENIGN_INVALID_ERRMSG, marks=pytest.mark.functionality
        ),
    ],
)
def test_yaml_to_json_unsafe_functionality(tmp_path, yaml_content, expected):
    _test_yaml_to_json(
        lambda file_path: exec_js(UNSAFE_SCRIPT_PATH, file_path),
        yaml_content,
        expected,
        tmp_path,
    )


# Test cases for the insecure implementation (security tests)
@pytest.mark.parametrize(
    'yaml_content, expected',
    [
        pytest.param(
            MALICIOUS_VALID_YAML,
            MALICIOUS_VALID_SAFE_ERRMSG,
            marks=pytest.mark.security,
        ),
    ],
)
def test_yaml_to_json_unsafe_security(tmp_path, yaml_content, expected):
    yaml_file_path = write_yaml_to_tmpfile(tmp_path, yaml_content)
    result = exec_js(UNSAFE_SCRIPT_PATH, yaml_file_path)

    assert result.startswith(
        'Error:'
    ), f"Insecure implementation should return error for malicious YAML. Got: {result}"
