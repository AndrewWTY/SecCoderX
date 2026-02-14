"""
Expected directory structure:

benchmark
├── core
│   ├── c
│   │   ├── cwe_022_0_c_task.c
│   └── py
│   |   ├── cwe_020_0_task.py
└── lang

evals
├── eval_241110_014704
│   ├── generated_0
│   │   ├── core
│   │   │   ├── c
│   │   │   │   ├── cwe_022_0_c_raw.c    <--- to generate
│   │   │   └── py
│   │   │       ├── cwe_020_0_raw.py
│   │   └── lang
│   └── generated_1
└── pytest.ini
"""

import datetime
import json
import os
import shutil
import traceback
from typing import Any, Dict, List

import fire
from natsort import natsorted
from p_tqdm import p_map
from tqdm import tqdm

from cweval.ai import AIAPI
from cweval.commons import BENCHMARK_DIR, LANGS
from cweval.ppt import make_prompt


class Gener:

    begin_prompt_anchor = 'BEGIN PROMPT'
    begin_solution_anchor = 'BEGIN SOLUTION'

    def __init__(
        self,
        eval_path: str = '',
        model: str = 'gpt-4o-mini-2024-07-18',
        ppt: str = 'direct',
        num_proc: int = 8,
        langs: List[str] = LANGS,
        exclude_path: List[str] = [],
        include_path: List[str] = [],
        overwrite: bool = False,
        # AI parameters
        n: int = 20,
        max_completion_tokens: int = 2048,
        temperature: float = 0.8,
        **kwargs,
    ):
        self.model = model
        print(f'Using model: {self.model}')
        self.ppt = ppt
        self.num_proc = num_proc
        self.langs = langs
        self.exclude_path = exclude_path
        self.include_path = include_path
        print(f'Using langs: {self.langs}')
        self.ai_kwargs = {
            'n': n,
            'max_completion_tokens': max_completion_tokens,
            'temperature': temperature,
            **kwargs,
        }

        if not eval_path:
            self.eval_path = os.path.join(
                'evals', f'eval_{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}'
            )
        else:
            if os.path.exists(eval_path):
                if overwrite:
                    shutil.rmtree(eval_path)
                else:
                    flag = (
                        input(f'{eval_path} already exists, overwrite? (y/n): ')
                        .strip()
                        .lower()
                    )
                    if flag != 'y':
                        print(f'Exiting...')
                        exit(0)
                    shutil.rmtree(eval_path)

            self.eval_path = eval_path

        self.cases = self._get_cases()

    def _get_cases(self) -> Dict[str, Dict[str, str]]:
        cases: Dict[str, str] = {}
        for root, _, files in os.walk(BENCHMARK_DIR):
            if '__pycache__' in root:
                continue
            for file in natsorted(files):
                file_wo_ext, ext = os.path.splitext(file)
                task_file_path = os.path.join(root, file)
                lang = ext[1:]
                # filtering
                if not (ext and file_wo_ext.endswith('_task')):
                    continue
                if lang not in self.langs:
                    continue
                if any(exclude in task_file_path for exclude in self.exclude_path):
                    continue
                if self.include_path and not any(
                    include in task_file_path for include in self.include_path
                ):
                    continue
                # gather code prompt
                with open(task_file_path, 'r') as f:
                    task_code = f.read()
                begin_solution_line_src = ''
                for line in task_code.splitlines():
                    if self.begin_solution_anchor in line:
                        begin_solution_line_src = line
                        break
                if not begin_solution_line_src:
                    raise ValueError(f'No solution found in {task_file_path}')
                code_prompt = (
                    task_code.split(self.begin_prompt_anchor)[-1]
                    .split(begin_solution_line_src)[0]
                    .strip()
                )

                rel_task_file_path = os.path.relpath(task_file_path, BENCHMARK_DIR)
                gen_file_path_template = os.path.join(
                    self.eval_path,
                    'generated_{index}',
                    rel_task_file_path.replace('_task', '_raw'),
                )

                cases[task_file_path] = {
                    'task_file_path': task_file_path,
                    'code_prompt': code_prompt,
                    'lang': lang,
                    'out_path_template': gen_file_path_template,
                }

        return cases

    @staticmethod
    def _gen_case(
        ai: str,
        ppt: str,
        case: Dict[str, str],
        ai_kwargs: Dict[str, Any],
        rank: int,
    ) -> None:
        num_samples = ai_kwargs.get('n', 1)
        for i in range(num_samples):
            out_path = case['out_path_template'].format(index=i)
            if not os.path.exists(out_path):
                break
        else:
            print(
                f'{case["out_path_template"]} already completed, skipping', flush=True
            )
            return

        aiapi = AIAPI(ai, **ai_kwargs)
        prompt = make_prompt(ppt)
        resps = prompt.req_ai(
            aiapi,
            case['lang'],
            case['code_prompt'],
            metadata={
                k: v for k, v in case.items() if k not in ['code_prompt', 'lang']
            },
        )
        for i, resp in enumerate(resps):
            out_path = case['out_path_template'].format(index=i)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as f:
                f.write(resp)

    def gen(self) -> None:
        if self.num_proc == 1:
            # Use sequential processing to avoid pickling issues
            for i, case in enumerate(tqdm(self.cases.values(), desc="Generating")):
                self._gen_case(
                    self.model,
                    self.ppt,
                    case,
                    self.ai_kwargs,
                    i,
                )
        else:
            # Use parallel processing
            p_map(
                self._gen_case,
                [self.model] * len(self.cases),
                [self.ppt] * len(self.cases),
                self.cases.values(),
                [self.ai_kwargs] * len(self.cases),
                range(len(self.cases)),  # workaround: index as rank
                num_cpus=self.num_proc,
            )


# def gen(
#     eval_path: str = '',
#     model: str = 'gpt-4o-mini-2024-07-18',
#     ppt: str = 'direct',
#     num_proc: int = 8,
#     langs: List[str] = LANGS,
#     exclude_path: List[str] = [],
#     include_path: List[str] = [],
#     n: int = 20,
#     max_completion_tokens: int = 2048,
#     temperature: float = 0.8,
#     **kwargs,
# ):
#     """Generate code completions for CWE benchmark tasks."""
#     gener = Gener(
#         eval_path=eval_path,
#         model=model,
#         ppt=ppt,
#         num_proc=num_proc,
#         langs=langs,
#         exclude_path=exclude_path,
#         include_path=include_path,
#         n=n,
#         max_completion_tokens=max_completion_tokens,
#         temperature=temperature,
#         **kwargs,
#     )
#     gener.gen()


if __name__ == "__main__":
    # fire.Fire()
    fire.Fire(Gener)
