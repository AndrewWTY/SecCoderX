#!/bin/bash

# PurpleLlama CybersecurityBenchmarks Batch Evaluation Script
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
GPU_ID=6
PORT=8032

# Benchmark settings
BENCHMARK="instruct"
PROMPT_FILE="instruct.json"  # Prompt file with reference codes

# Judge batch settings
JUDGE_API_KEY="YOUR_GOOGLE_API_KEY"
JUDGE_LLM="gemini-2.5-flash"

# Output prefix (leave empty for no prefix)
# Example: OUTPUT_PREFIX="exp1_" will create files like "exp1_model_name_instruct_responses.json"
OUTPUT_PREFIX="exp1_"

#==========================================
# END OF CONFIGURATION
#==========================================

# Base directory for PurpleLlama
PURPLELLAMA_DIR="/path/to/project/PurpleLlama"
DATASETS="$PURPLELLAMA_DIR/CybersecurityBenchmarks/datasets"

# Check that at least one model is specified
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Error: No models specified. Edit the MODELS array in the script."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p "${PURPLELLAMA_DIR}/logs"

# Change to PurpleLlama root directory
cd "$PURPLELLAMA_DIR"

# Activate mamba environment for vLLM serving
eval "$(mamba shell hook --shell bash)"
mamba activate verl

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

# Function to select reference model based on model name
select_reference_model() {
    local model_name="$1"
    local model_lower=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')

    if [[ "$model_lower" == *"codellama"* ]]; then
        echo "codellama-7b-instruct-hf"
    elif [[ "$model_lower" == *"qwen"* ]]; then
        if [[ "$model_lower" == *"7b"* ]]; then
            echo "qwen2-5-coder-7b-instruct"
        else
            # Default to 3b for qwen models
            echo "qwen2-5-coder-3b-instruct"
        fi
    else
        # Default fallback
        echo "qwen2-5-coder-3b-instruct"
    fi
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

# Function to stop vLLM server
stop_vllm_server() {
    local pid="$1"
    local log_file="$2"
    
    echo "Stopping vLLM server (PID: $pid)..." | tee -a "$log_file"
    if [ ! -z "$pid" ] && kill -0 $pid 2>/dev/null; then
        kill $pid 2>/dev/null || true
        # Wait a bit for graceful shutdown
        sleep 2
        # Force kill if still running
        if kill -0 $pid 2>/dev/null; then
            kill -9 $pid 2>/dev/null || true
        fi
    fi
}

# Print summary
echo "=========================================="
echo "PurpleLlama Batch Evaluation Script"
echo "=========================================="
echo "GPU ID: $GPU_ID"
echo "Port: $PORT"
echo "Benchmark: $BENCHMARK"
echo "Output prefix: ${OUTPUT_PREFIX:-<none>}"
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
    
    # Create unique timestamp for this model
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${PURPLELLAMA_DIR}/logs/${OUTPUT_PREFIX}${MODEL_NAME}_${TIMESTAMP}.txt"
    VLLM_LOG_FILE="${PURPLELLAMA_DIR}/logs/${OUTPUT_PREFIX}${MODEL_NAME}_${TIMESTAMP}_vllm.txt"
    RESPONSE_PATH="${DATASETS}/${OUTPUT_PREFIX}${MODEL_NAME}_${BENCHMARK}_responses.json"
    STAT_PATH="${DATASETS}/${OUTPUT_PREFIX}${MODEL_NAME}_${BENCHMARK}_stat.json"
    
    echo "=========================================="
    echo "[$((MODEL_IDX+1))/${#MODELS[@]}] Evaluating: $MODEL_NAME"
    echo "=========================================="
    echo "Model Path: $MODEL_PATH"
    echo "Log File: $LOG_FILE"
    echo "Response Path: $RESPONSE_PATH"
    echo "Stat Path: $STAT_PATH"
    echo ""
    
    # Start logging for this model
    echo "==========================================" | tee "$LOG_FILE"
    echo "Evaluation started at $(date)" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Model Path: $MODEL_PATH" | tee -a "$LOG_FILE"
    echo "Model Name: $MODEL_NAME" | tee -a "$LOG_FILE"
    echo "GPU ID: $GPU_ID" | tee -a "$LOG_FILE"
    echo "Port: $PORT" | tee -a "$LOG_FILE"
    echo "Benchmark: $BENCHMARK" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Reactivate mamba environment for vLLM serving
    mamba activate verl
    
    # Start vLLM server in the background
    echo "Starting vLLM server..." | tee -a "$LOG_FILE"
    echo "vLLM logs will be written to: $VLLM_LOG_FILE" | tee -a "$LOG_FILE"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve "$MODEL_PATH" \
        --trust-remote-code \
        --tensor-parallel-size 1 \
        --max-model-len 16384 \
        --disable-log-requests \
        --port $PORT \
        --served-model-name "$MODEL_NAME" \
        > "$VLLM_LOG_FILE" 2>&1 &
    
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
    echo "Running CybersecurityBenchmarks..." | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    
    # Activate virtual environment and run benchmark
    source ~/.venvs/CybersecurityBenchmarks/bin/activate

    # Select reference model based on model name
    REFERENCE_MODEL=$(select_reference_model "$MODEL_NAME")
    echo "Selected reference model: $REFERENCE_MODEL" | tee -a "$LOG_FILE"

    if ! python3 -m CybersecurityBenchmarks.benchmark.run \
        --benchmark=$BENCHMARK \
        --prompt-path="$DATASETS/$BENCHMARK/$PROMPT_FILE" \
        --response-path="$RESPONSE_PATH" \
        --stat-path="$STAT_PATH" \
        --llm-under-test="OPENAI::${MODEL_NAME}::sk-xxxxxx::http://0.0.0.0:${PORT}/v1" \
        --run-llm-in-parallel \
        --judge-llm="GOOGLEGENAI::${JUDGE_LLM}" \
        --judge-batch \
        --judge-api-key="$JUDGE_API_KEY" \
        --reference-model="$REFERENCE_MODEL" \
        --judge-batch-auto 
        2>&1 | tee -a "$LOG_FILE"; then
        echo "Benchmark failed!" | tee -a "$LOG_FILE"
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
    echo "Response saved to: $RESPONSE_PATH" | tee -a "$LOG_FILE"
    echo "Stats saved to: $STAT_PATH" | tee -a "$LOG_FILE"
    
    # Stop vLLM server for this model
    stop_vllm_server "$VLLM_PID" "$LOG_FILE"
    
    SUCCESS_MODELS+=("$MODEL_NAME")
    
    echo ""
    echo "Finished evaluating $MODEL_NAME"
    echo "Log saved to: $LOG_FILE"
    echo ""
    
    # Small delay before next model
    sleep 5
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

