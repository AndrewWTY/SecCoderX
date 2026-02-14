import os
import re

import fire
import pandas as pd
from natsort import natsorted

from cweval.commons import exec_cmd_shell

# Raw log data
LOG_DATA = """
================
pass@1  core/c
functional@1    64.10
secure@1        31.62
functional_secure@1     29.91
================
================
pass@3  core/c
functional@3    79.49
secure@3        43.59
functional_secure@3     38.46
================
================
pass@10 core/c
functional@10   100.00
secure@10       100.00
functional_secure@10    100.00
================
================
pass@1  core/cpp
functional@1    61.40
secure@1        33.33
functional_secure@1     31.58
================
================
pass@3  core/cpp
functional@3    73.68
secure@3        42.11
functional_secure@3     36.84
================
================
pass@10 core/cpp
functional@10   100.00
secure@10       100.00
functional_secure@10    100.00
================
================
pass@1  core/go
functional@1    63.16
secure@1        36.84
functional_secure@1     35.09
================
================
pass@3  core/go
functional@3    78.95
secure@3        47.37
functional_secure@3     47.37
================
================
pass@10 core/go
functional@10   100.00
secure@10       100.00
functional_secure@10    100.00
================
================
pass@1  core/py
functional@1    87.18
secure@1        52.56
functional_secure@1     50.00
================
================
pass@3  core/py
functional@3    92.31
secure@3        53.85
functional_secure@3     53.85
================
================
pass@10 core/py
functional@10   100.00
secure@10       100.00
functional_secure@10    100.00
================
================
pass@1  core/js
functional@1    83.33
secure@1        59.09
functional_secure@1     59.09
================
================
pass@3  core/js
functional@3    90.91
secure@3        72.73
functional_secure@3     72.73
================
================
pass@10 core/js
functional@10   100.00
secure@10       100.00
functional_secure@10    100.00
================
================
pass@1  lang/c
functional@1    96.97
secure@1        75.76
functional_secure@1     75.76
================
================
pass@3  lang/c
functional@3    100.00
secure@3        81.82
functional_secure@3     81.82
================
================
pass@10 lang/c
functional@10   100.00
secure@10       100.00
functional_secure@10    100.00
================
================
pass@1  all
functional@1    75.78
secure@1        46.44
functional_secure@1     45.01
================
================
pass@3  all
functional@3    86.32
secure@3        55.56
functional_secure@3     53.85
================
================
pass@10 all
functional@10   100.00
secure@10       100.00
functional_secure@10    100.00
"""


def table_report(input_path: str = '', return_df: bool = False) -> pd.DataFrame | None:
    if not input_path:
        log_data = LOG_DATA
    else:
        with open(input_path, 'r') as f:
            log_data = f.read()

    # Initialize storage for table data
    table_data = {}

    # Regular expressions for parsing
    section_regex = r"pass@(\d+)\s+([\w/]+)"
    metric_regex = r"(functional|secure|functional_secure)@(\d+)\s+([\d.]+)"

    # Parse the log data
    sections = log_data.strip().split("================\n")
    for section in sections:
        # Find language and pass@N
        section_match = re.search(section_regex, section)
        if not section_match:
            continue
        pass_n, language = section_match.groups()

        # Find each metric in the section
        metrics = re.findall(metric_regex, section)
        for metric_type, n, value in metrics:
            metric_name = f"{metric_type}@{n}"
            if metric_name not in table_data:
                table_data[metric_name] = {}
            table_data[metric_name][language] = float(value)

    # Convert to a pandas DataFrame for a table format
    df = pd.DataFrame(table_data).T
    df.index.name = "Metric"
    df.fillna("-", inplace=True)  # Fill missing entries with "-"
    dfp = df.T[
        filter(
            lambda x: x in df.T,
            [
                'functional@1',
                'functional@10',
                'functional@50',
                'secure@1',
                'secure@10',
                'secure@50',
                'functional_secure@1',
                'functional_secure@10',
                'functional_secure@50',
            ],
        )
    ]

    print(dfp)
    print(dfp.to_csv())
    # print csv
    # print(df.to_csv())
    if return_df:
        return df


def check_res():
    evals_dir = 'evals'

    model_dfs = {}

    for eval_job in natsorted(os.listdir(evals_dir)):
        # eval_4omini_t8
        model = '_'.join(eval_job.split('_')[1:-1])
        tstr = eval_job.split('_')[-1]
        eval_path = os.path.join(evals_dir, eval_job)  # evals/eval_4omini_t8
        res_json_path = os.path.join(eval_path, 'res_all.json')
        if os.path.exists(res_json_path):
            print(eval_job)


def merge_report():
    evals_dir = 'evals'

    model_dfs = {}

    for eval_job in natsorted(os.listdir(evals_dir)):
        # eval_4omini_t8
        model = '_'.join(eval_job.split('_')[1:-1])
        tstr = eval_job.split('_')[-1]
        eval_path = os.path.join(evals_dir, eval_job)  # evals/eval_4omini_t8
        res_json_path = os.path.join(eval_path, 'res_all.json')
        if os.path.exists(res_json_path):
            # python cweval/evaluate.py report_pass_at_k --eval_path evals/eval_4omini_t8 | tee evals/eval_4omini_t8/report.log
            cmd = f"python cweval/evaluate.py report_pass_at_k --eval_path {eval_path} | tee {eval_path}/report.log"
            print(cmd, flush=True)
            exec_cmd_shell(cmd)
            df = table_report(f"{eval_path}/report.log", return_df=True)
            if model not in model_dfs:
                model_dfs[model] = {}
            model_dfs[model][tstr] = df

    print(f'models: {model_dfs.keys()}', flush=True)
    model_max_df = {}

    for model, t2df in model_dfs.items():
        # merge tX
        t_dfs = [df for tstr, df in t2df.items() if tstr.startswith('t')]
        if not all(df.shape == t_dfs[0].shape for df in t_dfs):
            from IPython import embed

            embed()
        assert all(
            df.shape == t_dfs[0].shape for df in t_dfs
        ), f"All dataframes must have the same shape: {model = } , {t2df.keys() = }"
        max_t_df = pd.concat(t_dfs).groupby(level=0).max()
        model_max_df[model] = max_t_df

    model_all_df = {
        model: df.T.loc[['all']].rename(index={'all': model})
        for model, df in model_max_df.items()
    }
    # add greedy
    # from IPython import embed; embed()
    for model, df in model_all_df.items():
        if 'g' not in model_dfs[model]:
            df.insert(0, 'functional@1*', 0)
            df.insert(1, 'secure@1*', 0)
            df.insert(2, 'functional_secure@1*', 0)
        else:
            gdf = model_dfs[model]['g'].T.loc[['all']].rename(index={'all': model})
            df.insert(0, 'functional@1*', gdf['functional@1'][model])
            df.insert(1, 'secure@1*', gdf['secure@1'][model])
            df.insert(2, 'functional_secure@1*', gdf['functional_secure@1'][model])

    all_merged_df = pd.concat(model_all_df.values())

    # all_merged_df = all_merged_df[
    #     [
    #         'functional@1*',
    #         'secure@1*',
    #         'functional_secure@1*',
    #         'functional@1',
    #         'secure@1',
    #         'functional_secure@1',
    #         'functional@10',
    #         'secure@10',
    #         'functional_secure@10',
    #         'functional@50',
    #         'secure@50',
    #         'functional_secure@50',
    #     ]
    # ]
    all_merged_df = all_merged_df[
        [
            'functional@1*',
            'functional@1',
            'functional@10',
            'functional@50',
            'secure@1*',
            'secure@1',
            'secure@10',
            'secure@50',
            'functional_secure@1*',
            'functional_secure@1',
            'functional_secure@10',
            'functional_secure@50',
        ]
    ]

    # add greedy

    print(f'\n\n========================================\n')
    print(all_merged_df)
    print(all_merged_df.to_csv())
    # from IPython import embed; embed()


if __name__ == "__main__":
    fire.Fire()
