#!/bin/bash
eval "$(mamba shell hook --shell bash)"
mamba activate cweval
CWEVAL_DIR="/home/wutianyi/CWEval"
cd "$CWEVAL_DIR"
export PYTHONPATH=$PYTHONPATH:$(pwd)

# export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
# python cweval/generate.py gen --n 1 --temperature 0 --num_proc 16 --eval_path evals/eval_gpt4_1_t0_n1 --model gpt-4.1

# # Gemini through Google AI Studio
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
python cweval/generate.py gen --n 1 --temperature 0 --num_proc 16 --eval_path evals/eval_gflash2_5_flash_t0_n1 --model gemini/gemini-2.5-flash
python cweval/evaluate.py pipeline --eval_path evals/eval_gflash2_5_flash_t0_n1 --num_proc 20 --docker False
