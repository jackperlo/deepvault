#!/bin/bash

# Configuration
REPO_ID="jack-perlo/Lenet-5"
BASE_DIR="../../models_data"

# Prompt user for variant
echo "Choose the LeNet-5 model variant to download (fp32, uint8, int8). Enter the corresponding number:"
select variant in "fp32" "uint8" "int8"; do
  if [[ -n "$variant" ]]; then
    break
  else
    echo "Invalid selection."
  fi
done

# Set file extension and filename based on variant
if [ "$variant" = "fp32" ]; then
    MODEL_FILENAME="lenet5_fp32_mnist.keras"
else
    MODEL_FILENAME="lenet5_${variant}_mnist.tflite"
fi

# Create target directory
TARGET_DIR="${BASE_DIR}/lenet5"
mkdir -p "$TARGET_DIR"

echo "🚧 Downloading 'Lenet5_${variant}' model from Hugging Face Hub into '${TARGET_DIR}'..."

# Call Python to download and move the model
python3 - <<EOF
from huggingface_hub import hf_hub_download
import shutil
import os

repo_id = "$REPO_ID"
filename = f"$MODEL_FILENAME"
local_dir = "$TARGET_DIR"

# Download the model
local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")

# Move to the desired folder
shutil.copy(local_path, os.path.join(local_dir, "$MODEL_FILENAME"))

print(f"✅ Model downloaded and saved to: {local_dir}/$MODEL_FILENAME")
EOF
