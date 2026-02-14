#!/bin/bash
set -ex
python cweval/commons.py compile_all_in --path benchmark
pytest benchmark -x -n 24
