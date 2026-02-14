import multiprocessing as mp
import os
import re
import subprocess
import tempfile
from typing import Any, Callable, List, Tuple

import fire
import numpy as np
from natsort import natsorted

LANGS_COMPILE = ['c', 'cpp', 'go']
LANGS_RUN = ['py', 'js']
LANGS = LANGS_COMPILE + LANGS_RUN

COMPILE_DIR = 'compiled'
BENCHMARK_DIR = 'benchmark'


C_HEADERS = '''#include <archive_entry.h>
#include <jwt.h>
#include <errno.h>
#include <sys/wait.h>
#include <libxml/xpath.h>
#include <sqlite3.h>
#include <archive.h>
#include <ctype.h>
#include <libxml/parser.h>
#include <openssl/evp.h>
#include <openssl/des.h>
#include <openssl/aes.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <openssl/md5.h>
#include <stdlib.h>
#include <libgen.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <openssl/sha.h>
#include <regex.h>
#include <unistd.h>
#include <openssl/rand.h>
#include <archive.h>
#include <archive_entry.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <jwt.h>
#include <libgen.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <openssl/aes.h>
#include <openssl/des.h>
#include <openssl/evp.h>
#include <openssl/md5.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <regex.h>
#include <sqlite3.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
'''


CPP_HEADERS = '''#include <iostream>
#include <string>
#include <cstring>
#include <cctype>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <unistd.h>
#include <filesystem>
#include <archive.h>
#include <archive_entry.h>
#include <memory>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <map>
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <cstdio>
#include <openssl/aes.h>
#include <openssl/des.h>
#include <openssl/rand.h>
#include <tuple>
#include <stdexcept>
#include <jwt.h>
#include <libxml/parser.h>
#include <libxml/xpath.h>
#include <sys/stat.h>
#include <utility>
#include <regex>
#include <sqlite3.h>
'''


PY_HEADERS = '''import hashlib
import html
import tarfile
from urllib.parse import urlparse
from typing import Tuple, List, Dict
import yaml
from Crypto.Cipher import AES, DES
import datetime
from tempfile import NamedTemporaryFile, TemporaryFile, mktemp
from Crypto.Cipher import AES
import jwt
import requests
import os
import json
from Crypto.PublicKey import DSA
from Crypto.PublicKey import RSA
from yaml import Loader
import argon2
import re
import subprocess
import sqlite3
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
from lxml import etree
from Crypto.Cipher import AES
from Crypto.Cipher import AES, DES
from Crypto.PublicKey import DSA
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad
from lxml import etree
from tempfile import NamedTemporaryFile, TemporaryFile, mktemp
from urllib.parse import urlparse
from yaml import Loader
import argon2
import datetime
import hashlib
import html
import json
import jwt
import os
import re
import requests
import sqlite3
import subprocess
import tarfile
import yaml
'''


def complete_code(code: str, lang: str) -> str:
    if lang == 'c':
        return C_HEADERS + code
    elif lang == 'cpp':
        return CPP_HEADERS + code
    elif lang == 'py':
        return PY_HEADERS + code
    elif lang == 'go':
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(code)
            f.flush()
            filename = f.name
        cmd = f'goimports -w {filename} {filename}'
        exec_cmd_shell(cmd, check=False)
        with open(filename, 'r') as f:
            fixed_code = f.read()
        os.remove(filename)
        return fixed_code
    else:
        return code


def get_code_from(
    msg: str,
    only_last: bool = False,
    only_first: bool = False,
    add_new_line: bool = False,
) -> str:

    assert not (
        only_last and only_first
    ), '`only_last` and `only_first` cannot be both True'
    tail = '\n' if add_new_line else ''
    code_blocks: List[str] = []

    # Regex pattern to match fenced code blocks:
    # - Supports ``` and ~~~ fences
    # - Handles optional language specifier (e.g., ```python, ```js)
    # - Handles leading whitespace/indentation
    # - Non-greedy match for content
    pattern = r'^\s*(?:```|~~~)[\w]*\s*\n(.*?)^\s*(?:```|~~~)\s*$'

    matches = re.findall(pattern, msg, re.MULTILINE | re.DOTALL)

    if matches:
        for match in matches:
            # Strip trailing newline if present, then add tail
            code = match.rstrip('\n') + tail
            code_blocks.append(code)
            if only_first:
                return code_blocks[0]

    # Fallback: try to find inline code or code-like content
    if not code_blocks:
        # Try to match code blocks without proper closing (unclosed blocks)
        unclosed_pattern = r'^\s*(?:```|~~~)[\w]*\s*\n(.*?)(?=^\s*(?:```|~~~)|\Z)'
        unclosed_matches = re.findall(unclosed_pattern, msg, re.MULTILINE | re.DOTALL)
        if unclosed_matches:
            for match in unclosed_matches:
                code = match.rstrip('\n') + tail
                if code.strip():  # Only add non-empty blocks
                    code_blocks.append(code)
                    if only_first:
                        return code_blocks[0]

    # If still no code blocks found, return the entire message
    if not code_blocks:
        return msg

    if only_last:
        return code_blocks[-1]
    return '\n'.join(code_blocks)


def run_in_subprocess(func: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Run a function in a separate subprocess and return its result.

    Args:
        func: The function to run in the subprocess
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The return value from the function

    Example:
        def memory_intensive_function(x):
            # This will run in its own memory space
            large_list = [i for i in range(10**6)]
            return x * 2

        result = run_in_subprocess(memory_intensive_function, 5)
    """

    def wrapper(func, return_queue, *args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return_queue.put(result)
        except Exception as e:
            return_queue.put(e)

    # Create a queue to get the return value
    return_queue = mp.Queue()

    # Create and start the process
    process = mp.Process(
        target=wrapper, args=(func, return_queue) + args, kwargs=kwargs
    )
    process.start()

    # Wait for the process to complete
    process.join()

    # Get the result
    result = return_queue.get()

    # Check if the result is an exception
    if isinstance(result, Exception):
        raise result

    return result


def pass_at_k(n, c, k) -> float:
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def exec_cmd(
    cmd: List[str], check: bool = True, capture_output: bool = True
) -> Tuple[int, str, str]:
    assert isinstance(cmd, list)
    result = subprocess.run(cmd, capture_output=capture_output, text=True, check=check)
    return result.returncode, result.stdout, result.stderr


def exec_cmd_shell(
    cmd: str, check: bool = True, capture_output: bool = True
) -> Tuple[int, str, str]:
    assert isinstance(cmd, str)
    result = subprocess.run(
        cmd, capture_output=capture_output, text=True, check=check, shell=True
    )
    return result.returncode, result.stdout, result.stderr


def compile_c(
    src_path: str, compiled_path: str, check: bool = True
) -> Tuple[int, str, str]:
    lib_options = [
        '-lsqlite3',
        '-ljwt',
        # '-lcurl',
        '-lssl',
        '-lcrypto',
        '-larchive',
        '$(xml2-config --cflags --libs)',
    ]
    if 'lang/c' in src_path:
        exclude_patterns = [
            'lang/c/cwe_476_0_c_',
        ]
        if not any([pattern in src_path for pattern in exclude_patterns]):
            lib_options.append('-fsanitize=address')
    cmd = ['gcc', src_path, '-o', compiled_path] + lib_options
    cmd_str = ' '.join(cmd)
    returncode, stdout, stderr = exec_cmd_shell(cmd_str, check)
    if returncode != 0:
        print(f'Error compiling {src_path}:\n{stderr}', flush=True)
    return returncode, stdout, stderr


def compile_cpp(
    src_path: str, compiled_path: str, check: bool = True
) -> Tuple[int, str, str]:
    lib_options = [
        '-std=c++17',
        '-lsqlite3',
        '-ljwt',
        # '-lcurl',
        '-lssl',
        '-lcrypto',
        '-larchive',
        '$(xml2-config --cflags --libs)',
    ]
    if 'lang/cpp' in src_path:
        lib_options.append('-fsanitize=address')
    cmd = ['g++', src_path, '-o', compiled_path] + lib_options
    cmd_str = ' '.join(cmd)
    returncode, stdout, stderr = exec_cmd_shell(cmd_str, check)
    if returncode != 0:
        print(f'Error compiling {src_path}:\n{stderr}', flush=True)
    return returncode, stdout, stderr


def compile_go(
    src_path: str, compiled_path: str, check: bool = True
) -> Tuple[int, str, str]:
    cmd = ['go', 'build', '-o', compiled_path, src_path]
    cmd_str = ' '.join(cmd)
    returncode, stdout, stderr = exec_cmd_shell(cmd_str, check)
    if returncode != 0:
        print(f'Error compiling {src_path}:\n{stderr}', flush=True)
    return returncode, stdout, stderr


def compile_src(
    src_path: str, compiled_path: str, check: bool = True
) -> Tuple[int, str, str]:
    os.makedirs(os.path.dirname(compiled_path), exist_ok=True)
    lang = os.path.splitext(src_path)[1][1:]
    assert lang in LANGS_COMPILE, f'Unknown language for compile: {lang} for {src_path}'
    return {
        'c': compile_c,
        'cpp': compile_cpp,
        'go': compile_go,
    }[
        lang
    ](src_path, compiled_path, check)


def compile_list(
    src_path_list: List[str],
    compiled_path_list: List[str],
    check: bool = True,
    num_proc: int = 8,
) -> List[Tuple[int, str, str]]:
    # compiler sanity check
    cmd_checks = [
        'gcc --version',
        'g++ --version',
        'go version',
    ]
    for cmd_check in cmd_checks:
        returncode, stdout, stderr = exec_cmd_shell(cmd_check, check=True)

    assert len(src_path_list) == len(compiled_path_list)
    rets: List[Tuple[int, str, str]] = []
    if num_proc == 1:
        for src_path, compiled_path in zip(src_path_list, compiled_path_list):
            ret = compile_src(src_path, compiled_path, check)
            rets.append(ret)
    else:
        with mp.Pool(num_proc) as pool:
            rets = pool.starmap(
                compile_src,
                zip(src_path_list, compiled_path_list, [check] * len(src_path_list)),
            )
    return rets


def compile_all_in(
    path: str,
    check: bool = True,
    num_proc: int = 8,
) -> List[Tuple[int, str, str]]:
    src_path_list = []
    compiled_path_list = []
    if os.path.isfile(path):
        file_wo_ext, ext = os.path.splitext(path)
        if ext[1:] in LANGS_COMPILE:
            src_path_list.append(path)
            compiled_path = os.path.join(
                os.path.dirname(path), COMPILE_DIR, os.path.basename(file_wo_ext)
            )
            compiled_path_list.append(compiled_path)
    else:
        for root, _, files in os.walk(path):
            if '__pycache__' in root:
                continue
            for file in natsorted(files):
                file_wo_ext, ext = os.path.splitext(file)
                if ext[1:] in LANGS_COMPILE:
                    src_path = os.path.join(root, file)
                    compiled_path = os.path.join(root, COMPILE_DIR, file_wo_ext)
                    src_path_list.append(src_path)
                    compiled_path_list.append(compiled_path)

    return compile_list(src_path_list, compiled_path_list, check, num_proc)


if __name__ == '__main__':
    fire.Fire()
    # python cweval/commons.py compile_all_in --path benchmark
