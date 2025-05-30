#!/bin/bash

# Define the environment name and Python version
ENV_NAME="${1:-qanta-hf}"
PYTHON_VERSION="3.10"

# Check if conda is available
if ! command -v conda &>/dev/null; then
    echo "Conda could not be found. Please install it and try again."
    exit 1
fi
# Check if the environment already exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "The conda environment '$ENV_NAME' already exists."
    echo "What would you like to do?"
    echo "  [c] Continue using the existing environment"
    echo "  [e] Exit and retry after renaming the environment"
    echo "  [d] Delete the existing environment and create a new one"
    read -p "Enter your choice ([c]/e/d): " user_choice
    case "$user_choice" in
        [eE])
            echo "Exiting the script. Please invoke the script again with a different environment name."
            exit 1
            ;;
        [dD])
            echo "Deleting the environment '$ENV_NAME'..."
            conda env remove --name "$ENV_NAME"
            if [ $? -ne 0 ]; then
                echo "Failed to remove the environment. Exiting."
                exit 1
            fi
            ;;
        *)
            echo "Continuing with the existing environment '$ENV_NAME'."
            ;;
    esac
else
    # Create the conda environment
    echo "Creating a conda environment named $ENV_NAME with Python $PYTHON_VERSION"
    conda create -y --name $ENV_NAME python=$PYTHON_VERSION
fi

# Activate the environment
echo "Activating the conda environment: $ENV_NAME"

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
# Ensure the conda environment is activated
if [ $? -ne 0 ]; then
    echo "Failed to activate the conda environment: $ENV_NAME"
    exit 1
fi

# Install pytorch
uv pip install torch>=2.0.0

# Install dependencies
uv pip install -r requirements.txt