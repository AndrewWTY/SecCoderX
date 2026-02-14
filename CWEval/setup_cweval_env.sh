#!/bin/bash
set -e  # Exit on error

echo "🛡️  CWEval Environment Setup Script"
echo "=================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ============================================
# 1. Setup Mamba/Conda (if not exists)
# ============================================
echo "📦 Step 1: Checking for Mamba/Conda..."
if ! command -v mamba &> /dev/null && ! command -v conda &> /dev/null; then
    echo "⚠️  Mamba/Conda not found. Installing Miniforge3..."
    cd ~
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3
    rm Miniforge3-Linux-x86_64.sh

    # Initialize conda
    ~/miniforge3/bin/conda init bash
    ~/miniforge3/bin/conda init zsh 2>/dev/null || true

    # Source the conda setup
    source ~/miniforge3/bin/activate
    cd "$SCRIPT_DIR"
else
    echo "✅ Mamba/Conda found"
    # Try to activate conda if not already active
    if [ -z "$CONDA_PREFIX" ]; then
        if [ -f ~/miniforge3/bin/activate ]; then
            source ~/miniforge3/bin/activate
        elif [ -f ~/anaconda3/bin/activate ]; then
            source ~/anaconda3/bin/activate
        elif [ -f ~/miniconda3/bin/activate ]; then
            source ~/miniconda3/bin/activate
        fi
    fi
fi

# ============================================
# 2. Create Python Environment
# ============================================
echo ""
echo "🐍 Step 2: Creating cweval Python environment..."
if conda env list | grep -q "^cweval "; then
    echo "⚠️  Environment 'cweval' already exists. Removing it first..."
    conda env remove -n cweval -y
fi

if command -v mamba &> /dev/null; then
    mamba create -y -n cweval python=3.10
    # Initialize and activate using mamba
    eval "$(mamba shell hook --shell bash)"
    mamba activate cweval
else
    conda create -y -n cweval python=3.10
    # Activate using conda
    eval "$(conda shell.bash hook)"
    conda activate cweval
fi

echo "✅ Python environment created and activated"
python --version

# ============================================
# 3. Install C Dependencies
# ============================================
echo ""
echo "🔧 Step 3: Installing C dependencies..."

# Conda packages
if command -v mamba &> /dev/null; then
    mamba install -y libarchive zlib liblzma-devel
else
    conda install -y libarchive zlib liblzma-devel
fi

# System packages (requires sudo)
echo "Installing libjwt-dev (requires sudo)..."
if command -v apt &> /dev/null; then
    sudo apt update
    sudo apt install -y libjwt-dev
else
    echo "⚠️  apt not found. Please install libjwt-dev manually for your system."
fi

echo "✅ C dependencies installed"

# ============================================
# 4. Install JavaScript (Node.js) Dependencies
# ============================================
echo ""
echo "📦 Step 4: Installing JavaScript dependencies..."

# Install nvm if not exists
if [ ! -d "$HOME/.nvm" ]; then
    echo "Installing nvm..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

    # Add nvm to current shell
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
else
    echo "✅ nvm already installed"
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
fi

# Install and use Node.js LTS
echo "Installing Node.js LTS..."
nvm install --lts
nvm use --lts

# Install global npm packages
echo "Installing global npm packages..."
npm install -g escape-html node-rsa argon2 escape-string-regexp lodash js-yaml jsonwebtoken jsdom xpath sqlite3

# Export NODE_PATH for global modules
export NODE_PATH=$(npm root -g)

echo "✅ JavaScript dependencies installed"
node --version
npm --version

# ============================================
# 5. Install Golang Dependencies
# ============================================
echo ""
echo "🐹 Step 5: Installing Golang dependencies..."

if ! command -v go &> /dev/null; then
    echo "Installing Golang 1.23.3..."
    cd ~
    wget https://go.dev/dl/go1.23.3.linux-amd64.tar.gz
    sudo tar -C /usr/local -xzf go1.23.3.linux-amd64.tar.gz
    rm go1.23.3.linux-amd64.tar.gz
    cd "$SCRIPT_DIR"
else
    echo "✅ Golang already installed"
fi

# Add Go to PATH
export PATH=$PATH:/usr/local/go/bin

# Install Go tools
echo "Installing Go tools..."
go install golang.org/x/tools/cmd/goimports@latest
export PATH=$PATH:~/go/bin

# Download Go module dependencies
echo "Downloading Go module dependencies..."
cd "$SCRIPT_DIR"
go mod download

echo "✅ Golang dependencies installed"
go version

# ============================================
# 6. Install Python Dependencies
# ============================================
echo ""
echo "🐍 Step 6: Installing Python dependencies..."

# Make sure we're in the cweval environment
if command -v mamba &> /dev/null; then
    eval "$(mamba shell hook --shell bash)"
    mamba activate cweval
else
    eval "$(conda shell.bash hook)"
    conda activate cweval
fi

# Install core requirements
echo "Installing core requirements..."
pip install -r requirements/core.txt

# Install AI requirements
echo "Installing AI requirements..."
pip install -r requirements/ai.txt

# Install dev requirements (optional but recommended)
echo "Installing dev requirements..."
pip install -r requirements/dev.txt

# Setup pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install || echo "⚠️  pre-commit install failed, but continuing..."

echo "✅ Python dependencies installed"

# ============================================
# 7. Setup Environment Variables
# ============================================
echo ""
echo "🔧 Step 7: Setting up environment variables..."

# Create/update .env file if it doesn't exist
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    cat > "$SCRIPT_DIR/.env" << 'EOF'
# CWEval Environment Variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NODE_PATH=$(npm root -g)
export PATH=$PATH:/usr/local/go/bin
export PATH=$PATH:~/go/bin

# NVM Setup
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && . "$NVM_DIR/bash_completion"
EOF
    echo "✅ Created .env file"
else
    echo "✅ .env file already exists"
fi

# ============================================
# 8. Sanity Check
# ============================================
echo ""
echo "🧪 Step 8: Running sanity checks..."

# Source environment
source "$SCRIPT_DIR/.env"

# Compile reference solutions
echo "Compiling reference solutions..."
python cweval/commons.py compile_all_in --path benchmark/
if [ $? -eq 0 ]; then
    echo "✅ Compilation successful"
else
    echo "❌ Compilation failed"
    exit 1
fi

# Run tests (optional, can be commented out if too slow)
echo ""
echo "Would you like to run the full test suite? (y/n)"
read -t 10 -r RUN_TESTS || RUN_TESTS="n"
if [[ $RUN_TESTS =~ ^[Yy]$ ]]; then
    echo "Running tests (this may take a while)..."
    pytest benchmark/ -x -n 24
    if [ $? -eq 0 ]; then
        echo "✅ All tests passed"
    else
        echo "❌ Some tests failed"
    fi
else
    echo "⏭️  Skipping full test suite"
fi

# ============================================
# 9. Summary
# ============================================
echo ""
echo "=================================="
echo "✅ CWEval Environment Setup Complete!"
echo "=================================="
echo ""
echo "To use the environment:"
echo "  1. Activate conda: conda activate cweval"
echo "  2. Source environment: source .env"
echo ""
echo "Environment details:"
echo "  - Python: $(python --version 2>&1)"
echo "  - Node.js: $(node --version 2>&1)"
echo "  - Go: $(go version 2>&1)"
echo ""
echo "Next steps:"
echo "  - Set your API keys (e.g., export OPENAI_API_KEY=sk-xxxxx)"
echo "  - Generate LLM responses: python cweval/generate.py gen --model <model_name>"
echo "  - Evaluate responses: python cweval/evaluate.py pipeline --eval_path <path>"
echo ""
