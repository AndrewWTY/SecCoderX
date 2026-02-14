#!/bin/bash

# CWEval Batch Evaluation Script
# Edit the MODELS array below to specify which models to evaluate

set -o pipefail

#==========================================
# CONFIGURATION - Edit these values
#==========================================

# Models to evaluate (add/remove as needed)
MODELS=(
    "path/to/your/model/checkpoint"
    # "Qwen/Qwen2.5-Coder-7B-Instruct"
    # "meta-llama/CodeLlama-7b-Instruct-hf"
)
# GPU and server settings
GPU_ID=7
PORT=8195

# Generation settings
N_SAMPLES=1
TEMPERATURE=0
NUM_PROC=16

#==========================================
# END OF CONFIGURATION
#==========================================

# Check that at least one model is specified
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Error: No models specified. Edit the MODELS array in the script."
    exit 1
fi

# Base directory for CWEval
CWEVAL_DIR="/home/wutianyi/CWEval"

# Create logs directory if it doesn't exist
mkdir -p "${CWEVAL_DIR}/logs"

# Activate conda environment
eval "$(mamba shell hook --shell bash)"
mamba activate cweval

# Change to CWEval root directory
cd "$CWEVAL_DIR"

# Set Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)
export OPENAI_API_KEY=sk-xxxxxx

# Function to extract model name from path
extract_model_name() {
    local model_path="$1"
    local model_name

    if [[ "$model_path" != /* ]]; then
        # Not an absolute path - likely a HuggingFace model identifier
        # Replace / with _ (e.g., meta-llama/CodeLlama-7b-Instruct-hf -> meta-llama_CodeLlama-7b-Instruct-hf)
        model_name="${model_path//\//_}"
    else
        # Absolute path - extract from directory structure
        model_name=$(basename "$model_path")

        # Case 1: checkpoint-last or checkpoint-* -> use parent directory name
        # e.g., .../codellama-7b-instruct-safecoder/checkpoint-last -> codellama-7b-instruct-safecoder
        if [[ "$model_name" == checkpoint-* || "$model_name" == "checkpoint" ]]; then
            model_name=$(basename "$(dirname "$model_path")")
        # Case 2: huggingface/model/actor -> extract model_name_global_step_xxx
        # e.g., .../model_name/global_step_xxx/actor/huggingface
        elif [[ "$model_name" == "huggingface" || "$model_name" == "model" || "$model_name" == "actor" ]]; then
            local actor_dir=$(dirname "$model_path")           # .../actor
            local step_dir=$(dirname "$actor_dir")             # .../global_step_xxx
            local model_dir=$(dirname "$step_dir")             # .../model_name

            local step_name=$(basename "$step_dir")            # global_step_xxx
            local base_model_name=$(basename "$model_dir")     # model_name

            model_name="${base_model_name}_${step_name}"
        fi
    fi

    echo "$model_name"
}

# Function to check if vLLM server is ready
check_server_ready() {
    local port="$1"
    local max_attempts=60  # 5 minutes max (60 * 15 seconds)
    local attempt=0

    echo "Waiting for vLLM server to be ready..."

    # Wait at least 1 minute before checking
    echo "Initial wait of 1 minute for model loading..."
    sleep 60

    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "vLLM server is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        echo "Attempt $attempt/$max_attempts: Server not ready yet, waiting..."
        sleep 15
    done

    echo "Error: vLLM server failed to start within timeout"
    return 1
}

# Function to stop vLLM server and all its child processes
stop_vllm_server() {
    local pid="$1"
    local log_file="$2"

    echo "Stopping vLLM server (PID: $pid) and child processes..." | tee -a "$log_file"
    if [ ! -z "$pid" ] && kill -0 $pid 2>/dev/null; then
        # First, kill all child processes recursively
        pkill -TERM -P $pid 2>/dev/null || true
        # Kill the main process
        kill -TERM $pid 2>/dev/null || true
        # Wait a bit for graceful shutdown
        sleep 3
        # Force kill any remaining child processes
        pkill -9 -P $pid 2>/dev/null || true
        # Force kill the main process if still running
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid 2>/dev/null || true
        fi
        # Wait for GPU memory to be released
        sleep 2
    fi
}

# Print summary
echo "=========================================="
echo "CWEval Batch Evaluation Script"
echo "=========================================="
echo "GPU ID: $GPU_ID"
echo "Port: $PORT"
echo "N Samples: $N_SAMPLES"
echo "Temperature: $TEMPERATURE"
echo "Number of models: ${#MODELS[@]}"
echo "Models to evaluate:"
for i in "${!MODELS[@]}"; do
    echo "  $((i+1)). ${MODELS[$i]}"
done
echo "=========================================="
echo ""

# Track failed models
FAILED_MODELS=()
SUCCESS_MODELS=()

# Evaluate each model
for MODEL_IDX in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$MODEL_IDX]}"
    MODEL_NAME=$(extract_model_name "$MODEL_PATH")

    # Create naming suffix with temperature and samples
    NAMING_SUFFIX="temp_${TEMPERATURE}_samples_${N_SAMPLES}"
    LOG_FILE="${CWEVAL_DIR}/logs/${MODEL_NAME}_${NAMING_SUFFIX}.txt"
    EVAL_PATH="${CWEVAL_DIR}/evals/eval_${MODEL_NAME}_${NAMING_SUFFIX}"
    VLLM_LOG_FILE="${CWEVAL_DIR}/logs/${MODEL_NAME}_${NAMING_SUFFIX}_vllm.txt"

    echo "=========================================="
    echo "[$((MODEL_IDX+1))/${#MODELS[@]}] Evaluating: $MODEL_NAME"
    echo "=========================================="
    echo "Model Path: $MODEL_PATH"
    echo "Log File: $LOG_FILE"
    echo "Eval Path: $EVAL_PATH"
    echo ""

    # Start logging for this model
    echo "==========================================" | tee "$LOG_FILE"
    echo "Evaluation started at $(date)" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Model Path: $MODEL_PATH" | tee -a "$LOG_FILE"
    echo "Model Name: $MODEL_NAME" | tee -a "$LOG_FILE"
    echo "GPU ID: $GPU_ID" | tee -a "$LOG_FILE"
    echo "Port: $PORT" | tee -a "$LOG_FILE"
    echo "N Samples: $N_SAMPLES" | tee -a "$LOG_FILE"
    echo "Temperature: $TEMPERATURE" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # Start vLLM server in the background
    echo "Starting vLLM server..." | tee -a "$LOG_FILE"
    echo "vLLM logs will be written to: $VLLM_LOG_FILE" | tee -a "$LOG_FILE"

    # Check if model path contains "safecoder" (case-insensitive)
    if [[ "${MODEL_PATH,,}" == *"safecoder"* ]]; then
        echo "Base model detected (safecoder), using passthrough template" | tee -a "$LOG_FILE"
        CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve "$MODEL_PATH" \
            --trust-remote-code \
            --tensor-parallel-size 1 \
            --disable-log-requests \
            --max-model-len 16384 \
            --port $PORT \
            --served-model-name "$MODEL_NAME" \
            --chat-template "/home/wutianyi/PurpleLlama/passthrough_template.jinja" \
            > "$VLLM_LOG_FILE" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve "$MODEL_PATH" \
            --trust-remote-code \
            --tensor-parallel-size 1 \
            --disable-log-requests \
            --max-model-len 16384 \
            --port $PORT \
            --served-model-name "$MODEL_NAME" \
            > "$VLLM_LOG_FILE" 2>&1 &
    fi

    VLLM_PID=$!
    echo "vLLM server started with PID: $VLLM_PID" | tee -a "$LOG_FILE"

    # Wait for server to be ready
    if ! check_server_ready "$PORT" 2>&1 | tee -a "$LOG_FILE"; then
        echo "Failed to start vLLM server" | tee -a "$LOG_FILE"
        echo "Check vLLM logs at: $VLLM_LOG_FILE" | tee -a "$LOG_FILE"
        stop_vllm_server "$VLLM_PID" "$LOG_FILE"
        FAILED_MODELS+=("$MODEL_NAME")
        echo "Skipping to next model..." | tee -a "$LOG_FILE"
        sleep 5
        continue
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Running generation..." | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"

    if ! python cweval/generate.py gen \
        --n $N_SAMPLES \
        --temperature $TEMPERATURE \
        --num_proc $NUM_PROC \
        --eval_path "$EVAL_PATH" \
        --model "openai/$MODEL_NAME" \
        --api_base "http://localhost:${PORT}/v1" \
        --overwrite True \
        2>&1 | tee -a "$LOG_FILE"; then
        echo "Generation failed!" | tee -a "$LOG_FILE"
        stop_vllm_server "$VLLM_PID" "$LOG_FILE"
        FAILED_MODELS+=("$MODEL_NAME")
        echo "Skipping to next model..." | tee -a "$LOG_FILE"
        sleep 5
        continue
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Running evaluation..." | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"

    if ! python cweval/evaluate.py pipeline \
        --model "openai/$MODEL_NAME" \
        --eval_path "$EVAL_PATH" \
        --num_proc $NUM_PROC \
        --docker False \
        2>&1 | tee -a "$LOG_FILE"; then
        echo "Evaluation failed!" | tee -a "$LOG_FILE"
        stop_vllm_server "$VLLM_PID" "$LOG_FILE"
        FAILED_MODELS+=("$MODEL_NAME")
        echo "Skipping to next model..." | tee -a "$LOG_FILE"
        sleep 5
        continue
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Evaluation completed at $(date)" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"

    # Stop vLLM server for this model
    stop_vllm_server "$VLLM_PID" "$LOG_FILE"

    SUCCESS_MODELS+=("$MODEL_NAME")

    echo ""
    echo "Finished evaluating $MODEL_NAME"
    echo "Log saved to: $LOG_FILE"
    echo ""

    # Small delay before next model
    sleep 30
done

# Print summary
echo ""
echo "=========================================="
echo "Batch Evaluation Summary"
echo "=========================================="
echo "Total models: ${#MODELS[@]}"
echo "Successful: ${#SUCCESS_MODELS[@]}"
echo "Failed: ${#FAILED_MODELS[@]}"

if [ ${#SUCCESS_MODELS[@]} -gt 0 ]; then
    echo ""
    echo "Successful models:"
    for model in "${SUCCESS_MODELS[@]}"; do
        echo "  ✓ $model"
    done
fi

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo ""
    echo "Failed models:"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  ✗ $model"
    done
fi

echo "=========================================="
echo "All done!"
