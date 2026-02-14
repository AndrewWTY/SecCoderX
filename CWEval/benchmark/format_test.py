import os

import pytest
from natsort import natsorted

BENCHMARK_DIR = 'benchmark'
begin_prompt_anchor = 'BEGIN PROMPT'
begin_solution_anchor = 'BEGIN SOLUTION'
begin_entrypoint_anchor = 'BEGIN ENTRYPOINT'


def test_task_format():
    cwe_set = set()
    sce_set = set()
    task_set = set()

    for root, _, files in os.walk(BENCHMARK_DIR):
        if '__pycache__' in root:
            continue
        for file in natsorted(files):
            file_wo_ext, ext = os.path.splitext(file)
            if not (ext and file_wo_ext.endswith('_task')):
                continue
            # cwe_095_0_{lang}_task.{lang}
            cweid = '_'.join(file_wo_ext.split('_')[:2])
            sceid = '_'.join(file_wo_ext.split('_')[:3])
            task_name = file_wo_ext.split('_task')[0]
            cwe_set.add(cweid)
            sce_set.add(sceid)
            task_set.add(task_name)

            task_file_path = os.path.join(root, file)
            lang = ext[1:]
            with open(task_file_path, 'r') as f:
                task_code = f.read()

            assert (
                begin_solution_anchor in task_code
            ), f'No {begin_solution_anchor = } found in {task_file_path}'
            if lang != 'py':
                assert (
                    begin_prompt_anchor in task_code
                ), f'No {begin_prompt_anchor = } found in {task_file_path}'
                assert (
                    begin_entrypoint_anchor in task_code
                ), f'No {begin_entrypoint_anchor = } found in {task_file_path}'

    print(f'{len(cwe_set) = }', flush=True)
    print(f'{len(sce_set) = }', flush=True)
    print(f'{len(task_set) = }', flush=True)
