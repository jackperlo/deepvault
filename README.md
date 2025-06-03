# deepvault
# 🧠 Pre-trained Models and Datasets Collection

This repository brings together a curated collection of pre-trained models and datasets, created to cut through the noise of searching the internet for reliable and well-documented resources.  
My hope is that this grows into a robust pool of reusable, architecture-level models. **Contributions are very welcome!**

> If you've ever felt lost in the jungle of scattered links and incomplete documentation — this is for you. Enjoy!

---

### ✅ Supported Frameworks

- **Keras**
- **TensorFlow**

---

### 📊 What’s Included

The colormap below illustrates which pre-trained models are available for each dataset and in which precision format (`fp32`, `int8`, `uint8`).  
All models are hosted on the **Hugging Face Hub** for easy download and integration.

![Accuracy Heatmap](readme_assets/accuracy_map.png)

---

### 📦 How to Use

You have two main options:

#### 🔧 Option 1: Clone the Repository and mount the Docker Image (recommended for development/debugging)

  ```bash
  git clone https://github.com/jackperlo/deepvault.git
  
  cd deepvault
  ```

Mount the Docker Image:

  ```bash
  docker run --rm -it -v $(pwd):/deepvault -w /deepvault jackisback/deepvault:latest 
  ```

Use the structure and notebooks inside the repository to:

- Choose datasets and models
- Train or download pre-trained models
- Preprocess datasets
- Run evaluations

#### 🐳 Option 2: Pull the Docker Image (Quick start, no code modification)

If you don’t need to debug or modify the code, but just to execute it, pull the Docker image from Docker Hub:

  ```bash
  docker pull jackisback/deepvault:latest
  
  docker run --rm -it jackisback/deepvault:latest
  ```

This is the fastest way to access all scripts and resources in a clean, ready-to-use environment.

---

### 📚 Included Datasets

| Dataset        | Source                                                             |
|----------------|--------------------------------------------------------------------|
| `mnist`        | Provided via `tf.keras.datasets`                                   |
| `cifar10`      | Provided via `tf.keras.datasets`                                   |
| `gtsrb`        | Soon provided via HuggingFace                                      |
| `imagenet2012` | Hosted privately on Hugging Face due to licensing restrictions     |

**Note:**  
`imagenet2012` (ILSVRC 2012) is a large-scale image classification dataset. Its distribution is subject to specific academic/research use restrictions.  
To access this dataset, you must request permission from the official ImageNet website: [https://www.image-net.org/challenges/LSVRC/2012/](https://www.image-net.org/challenges/LSVRC/2012/).  
Only after approval can you legally download or use the dataset. For this reason, the hosted version is marked as private on Hugging Face [Models and Datasets repository](https://huggingface.co/jack-perlo).

---

### 🧠 Included Models

The following model architectures are supported:
1.  CNNs
  - `lenet5`
  - `efficientnetb0`
  - `mobilenet`
  - `mobilenetv2`
  - `vgg16`
  - `alexnet`
  - `resnet18`
  - `resnet34`
  - `resnet50`
  - `resnet152`
2. ViTs
  - `vit-b_16p_224` (Vision Transformer)

---

### 👣 Step-by-step Workflow

1. **Choose a Dataset**  
  Navigate to the folder of the dataset you’re interested in. (e.g., `Mnist`)

2. **Download or Train a Model**  
  Find the corresponding model folder (e.g., `Mnist/CNN/`)

  - **(a) Use a Custom Model**  
    Use the corresponding notebook to train or customize your model (e.g., `Mnist/CNN/lenet5.ipynb`). Once ready, save it in the desired format (FP32/INT8/UINT8) using the notebook.

  - **(b) Use a Pre-trained Model**  
    Run the provided `download_models.sh` shell script to download the desired model.

    ```bash
    bash download_models.sh
    ```

3. **Preprocess the Dataset**  
  Use the corresponding `<DS>_preprocessing.ipynb` notebook to preprocess the dataset images correctly and store them in the required format. (e.g., `Mnist/mnist_preprocessing.ipynb`)

4. **Evaluate the Model**  
  Run evaluation on a portion or the entirety of the preprocessed dataset using the corresponding `<DS>_models_evaluation.ipynb` and choosing the desired model inside the notebook cells. (e.g., `Mnist/mnist_models_evaluation.ipynb`)
