#!/bin/bash

# Define available datasets
datasets=("mnist" "imagenet2012" "cifar10")

#########################################
# Step 1: Select dataset
#########################################
echo "📦 Select the dataset:"
select dataset in "${datasets[@]}"; do
  if [[ -n "$dataset" ]]; then break; fi
  echo "❌ Invalid selection."
done

#########################################
# Step 2: Select model based on dataset
#########################################
case "$dataset" in
  "mnist")
    models=("lenet5")
    ;;
  "imagenet2012")
    models=("efficientnetb0" "mobilenetv1" "mobilenetv2" "resnet50" "resnet152" "vgg16" "vit_b16_p_224")
    ;;
  "cifar10")
    models=("alexnet" "efficientnetb0" "mobilenetv1" "mobilenetv2" "resnet50" "resnet152" "vgg16")
    ;;
  *)
    echo "❌ Unsupported dataset."
    exit 1
    ;;
esac

echo "🧠 Select model for $dataset:"
select model in "${models[@]}"; do
  if [[ -n "$model" ]]; then break; fi
  echo "❌ Invalid selection."
done

#########################################
# Step 3: Select precision
#########################################
case "$model" in
  "lenet5"|"efficientnetb0"|"mobilenetv1"|"mobilenetv2"|"vgg16"|"vit_b16_p_224"|"alexnet"|'resnet50'|"resnet152")
    precisions=("fp32" "uint8" "int8")
    ;;
  *)
    echo "❌ Unsupported model."
    exit 1
    ;;
esac

echo "🎯 Select precision for $model:"
select precision in "${precisions[@]}"; do
  if [[ -n "$precision" ]]; then break; fi
  echo "❌ Invalid selection."
done

#########################################
# Compose Hugging Face repo ID and file name
#########################################
# repo_id fallback
if [[ "$model" == "lenet5" ]]; then
  REPO_ID="jack-perlo/Lenet-5"
elif [[ "$model" == "resnet50" ]]; then
  REPO_ID="jack-perlo/ResNet50"
elif [[ "$model" == "resnet152" ]]; then
  REPO_ID="jack-perlo/ResNet152"
elif [[ "$model" == "efficientnetb0" ]]; then
  REPO_ID="jack-perlo/EfficientNetB0"
elif [[ "$model" == "mobilenetv1" ]]; then
  REPO_ID="jack-perlo/MobileNetV1"
elif [[ "$model" == "mobilenetv2" ]]; then
  REPO_ID="jack-perlo/MobileNetV2"
elif [[ "$model" == "vgg16" ]]; then
  REPO_ID="jack-perlo/VGG16"
elif [[ "$model" == "vit_b16_p_224" ]]; then
  REPO_ID="jack-perlo/Vit-B16-P-224"
elif [[ "$model" == "alexnet" ]]; then
  REPO_ID="jack-perlo/AlexNet"
else
  echo "❌ Unknown model. Exiting."
  exit 1
fi

# Determine filename
if [[ "$precision" == "fp32" ]]; then
  FILENAME="${model}_fp32_${dataset}.keras"
else
  FILENAME="${model}_${precision}_${dataset}.tflite"
fi

# Compose target directory
if [[ "$dataset" == "imagenet2012" ]]; then
  TARGET_DIR="./ImageNet2012/models_data/${model}"
elif [[ "$dataset" == "cifar10" ]]; then
  TARGET_DIR="./Cifar10/models_data/${model}"
elif [[ "$dataset" == "mnist" ]]; then
  TARGET_DIR="./Mnist/models_data/${model}"
else
  # Default case for other datasets
  echo "❌ Dataset chosen has no folder in this repository. Try again."
  exit 1
fi

mkdir -p "$TARGET_DIR"

#########################################
# Python download logic
#########################################
echo "🚧 Downloading model: $FILENAME from $REPO_ID into $TARGET_DIR ..."
python3 - <<EOF
from huggingface_hub import hf_hub_download
import shutil
import os

repo_id = "$REPO_ID"
filename = "$FILENAME"
target_dir = "$TARGET_DIR"

# Download from Hugging Face
local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")

# Copy to target dir
shutil.copy(local_path, os.path.join(target_dir, filename))
print(f"✅ Model saved to: {target_dir}/{filename}")
EOF
