#!/bin/bash

# Define available datasets
datasets=("mnist" "imagenet2012" "cifar10" "gtsrb")

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
    models=("efficientnetB0" "mobilenet" "mobilenetV2" "resnet50" "resnet152" "vgg16" "vit-b_16p_224")
    ;;
  "cifar10")
    models=("alexnet" "efficientnetB0" "mobilenet" "mobilenetV2" "resnet18" "resnet34" "resnet50" "resnet152" "vgg16")
    ;;
  "gtsrb")
    models=("mobilenet" "mobilenetV2" "resnet18" "resnet34")
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
  "lenet5"|"efficientnetB0"|"mobilenet"|"mobilenetV2"|"vgg16"|"alexnet"|'resnet18'|'resnet34'|'resnet50'|"resnet152"|"vit-b_16p_224")
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

############################################
# Compose Hugging Face repo ID and file name
############################################
if [[ "$model" == "lenet5" && "$dataset" == "mnist" ]]; then
  REPO_ID="jack-perlo/Lenet5-Mnist"
elif [[ "$model" == "mobilenet" && "$dataset" == "gtsrb" ]]; then
  REPO_ID="jack-perlo/MobileNet-Gtsrb"
elif [[ "$model" == "mobilenetv2" && "$dataset" == "gtsrb" ]]; then
  REPO_ID="jack-perlo/MobileNetV2-Gtsrb"
elif [[ "$model" == "resnet18" && "$dataset" == "gtsrb" ]]; then
  REPO_ID="jack-perlo/ResNet18-Gtsrb"
elif [[ "$model" == "resnet34" && "$dataset" == "gtsrb" ]]; then
  REPO_ID="jack-perlo/ResNet34-Gtsrb"
elif [[ "$model" == "alexnet" && "$dataset" == "cifar10" ]]; then
  REPO_ID="jack-perlo/AlexNet-Cifar10"
elif [[ "$model" == "resnet18" && "$dataset" == "cifar10" ]]; then
  REPO_ID="jack-perlo/ResNet18-Cifar10"
elif [[ "$model" == "resnet34" && "$dataset" == "cifar10" ]]; then
  REPO_ID="jack-perlo/ResNet34-Cifar10"
elif [[ "$model" == "resnet50" && "$dataset" == "cifar10" ]]; then
  REPO_ID="jack-perlo/ResNet50-Cifar10"
elif [[ "$model" == "resnet152" && "$dataset" == "cifar10" ]]; then
  REPO_ID="jack-perlo/ResNet152-Cifar10"
elif [[ "$model" == "efficientnetB0" && "$dataset" == "cifar10" ]]; then
  REPO_ID="jack-perlo/EfficientNetB0-Cifar10"
elif [[ "$model" == "mobilenet" && "$dataset" == "cifar10" ]]; then
  REPO_ID="jack-perlo/MobileNet-Cifar10"
elif [[ "$model" == "mobilenetV2" && "$dataset" == "cifar10" ]]; then
  REPO_ID="jack-perlo/MobileNetV2-Cifar10"
elif [[ "$model" == "vgg16" && "$dataset" == "cifar10" ]]; then
  REPO_ID="jack-perlo/Vgg16-Cifar10"
elif [[ "$model" == "resnet50" && "$dataset" == "imagenet2012" ]]; then
  REPO_ID="jack-perlo/ResNet50-ImageNet2012"
elif [[ "$model" == "resnet152" && "$dataset" == "imagenet2012" ]]; then
  REPO_ID="jack-perlo/ResNet152-ImageNet2012"
elif [[ "$model" == "efficientnetB0" && "$dataset" == "imagenet2012" ]]; then
  REPO_ID="jack-perlo/EfficientNetB0-ImageNet2012"
elif [[ "$model" == "mobilenet" && "$dataset" == "imagenet2012" ]]; then
  REPO_ID="jack-perlo/MobileNet-ImageNet2012"
elif [[ "$model" == "mobilenetV2" && "$dataset" == "imagenet2012" ]]; then
  REPO_ID="jack-perlo/MobileNetV2-ImageNet2012"
elif [[ "$model" == "vgg16" && "$dataset" == "imagenet2012" ]]; then
  REPO_ID="jack-perlo/Vgg16-ImageNet2012"
elif [[ "$model" == "vit-b_16p_224" && "$dataset" == "imagenet2012" ]]; then
  REPO_ID="jack-perlo/ViT-b_16p_224-Imagenet2012"
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
elif [[ "$dataset" == "gtsrb" ]]; then
  TARGET_DIR="./Gtsrb/models_data/${model}"
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
