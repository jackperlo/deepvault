import os
import pandas as pd
import torch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torchvision

from torch.utils.data import DataLoader



# determine how many batches (50k/DS_BATCH_SIZE) will be returned by the dataset
DS_BATCH_SIZE = 10000 

# VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
# !NOTE: TFLite, only first 10000 VALIDATION images are loaded! NO train images are available.
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# number of samples in the dataset
N_SAMPLES = 10000 

# VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
# !NOTE: PYTORCH, only first 2000 VALIDATION images are loaded! NO train images are available.
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# number of images loaded for the torch-imported models; 
# this value depends on the length of the file in the CSV_LABELS_PATH 
N_TORCH_IMPORTED_MODELS_LOADED_IMAGES = 2000

# path to the Imagenet validation set predicted labels file
TXT_LABELS_PATH = './datasets_data/ILSVRC2012/labels/val.txt'
CSV_LABELS_PATH = './datasets_data/ILSVRC2012/images/val.csv'
# path to the Imagenet image set file
IMAGES_DIR_PATH = './datasets_data/ILSVRC2012/images/'

"""
  TorchImageNet2012Dataset class is responsible for loading and 
  preprocessing the ImageNet validation dataset for the torch-trained models.
  It is a subclass of torch.utils.data.Dataset, allowing to generate custom dataset objects.
"""
class TorchImageNet2012Dataset(torch.utils.data.Dataset):
  def __init__(self, annotations_file, img_dir, transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    
    image = torchvision.io.read_image(img_path)
    # ensure 3 channels for consistency
    if image.shape[0] == 1:  
        image = image.repeat(3, 1, 1)  # Convert grayscale to RGB
    
    label = self.img_labels.iloc[idx, 1]
    
    if self.transform:
      image = self.transform(image)
    
    sample = {'image': image, 'label': label}
    return sample

"""
  ImageNetDataset class is responsible for loading and preprocessing the 
  ImageNet validation dataset.
"""
class ImageNetDataset:
  _mode = ''
  _framework = ''
  _type = ''

  """
  (private, static)
  Pre-process the .jpeg image in the 'torch' mode.
  This method scales pixels according to the per-model pre-processing
  input images function, sample-wise.

  Args:
    image_path: str
      The path of input image to be pre-processed.
    label: str
      The label of the input image.

  Returns:  
    tuple
      The tuple containing the pre-processed image 
      as (224,224,3) float32 scaled in the correct range, and
      the corresponding labels.
  """
  @staticmethod
  def _load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path) 
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
   
    if ImageNetDataset._mode == 'only_labels':
      # if only labels are needed, return the image as None
      return image, label
    if ImageNetDataset._mode == 'alexnet':
      image = tf.cast(image, tf.float32)
      image /= 127.5
      image -= 1.0
    elif ImageNetDataset._mode == 'efficientnetb0':
      image = tf.keras.applications.efficientnet.preprocess_input(image)
    elif ImageNetDataset._mode == 'mobilenet':
      image = tf.keras.applications.mobilenet.preprocess_input(image)
    elif ImageNetDataset._mode == 'mobilenetv2':
      image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    elif ImageNetDataset._mode == 'resnet50':
      image = tf.keras.applications.resnet50.preprocess_input(image)
    elif ImageNetDataset._mode == 'resnet152':
      image = tf.keras.applications.resnet.preprocess_input(image)
    elif ImageNetDataset._mode == 'vgg16':
      image = tf.keras.applications.vgg16.preprocess_input(image)
    elif ImageNetDataset._mode == 'ViT-b_16p':
      return image, label
    else:
      e = f"Invalid mode {ImageNetDataset._mode}, framework {ImageNetDataset._framework} not supported."
      raise ValueError(e)

    return image, label 

  """
  Load and return the custom torch dataset given the csv file 
  containing the <img_id,golden_label> pairs and the JPEG images directory.
  The trasnform parameter is the pre-processing function to be applied to the images.
  """
  @staticmethod
  def _load_torchvision_dataset():
    if ImageNetDataset._mode == 'resnet18':
      return TorchImageNet2012Dataset(
        annotations_file=CSV_LABELS_PATH,
        img_dir=IMAGES_DIR_PATH,
        transform=torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms()
      )
    else:
      e = f"Invalid mode {ImageNetDataset._mode}, framework {ImageNetDataset._framework} not supported."
      raise ValueError(e)
    
  """
  Load the custom torch dataset and return the contained images and labels as numpy arrays.
  """
  @staticmethod
  def _load_torchvision_dataset_as_numpy():
    dataset = ImageNetDataset._load_torchvision_dataset()
    dataloader = DataLoader(dataset, batch_size=N_TORCH_IMPORTED_MODELS_LOADED_IMAGES, shuffle=False)
    images = np.array([])
    labels = np.array([])
    for batch in dataloader:
      image = batch['image'].numpy()
      label = batch['label'].numpy()
      if len(images) == 0:
        images = image
        labels = label
      else:
        np.vstack(images, image)
        np.vstack(labels, label)
    return images, labels

  """
  (public, static)
  Load the tensor of image paths and labels from the ImageNet validation file
  which contains, per each line: (<IMG_0000X>.JPEG, <predicted_label>) and load
  the images from the disk populating a Dataset object.
  A tuple containing training and validation images, labels is returned.

  Args:
    mode: str
      The mode of the pre-processing. 
      Default is None.
    framework: str
      The framework for which compute the pre-processing. 
      Default is 'tensorflow'. 
      Supported are ['tensorflow', 'torch'].
    type: str
      The type to load the dataset.
      Default is None.
      Supported is 'numpy'.

  Returns:
    - If framework is 'tensorflow':
      tuple
        A tuple containing two tuples: 
        (training_images=None, training_labels=None),
        (validation images   , validation_labels).
    - If framework is 'torch':
      a torch.utils.data.Dataset object.
    - If framework is 'torch' and type is 'numpy':
      tuple
        A tuple containing two numpy.ndarrays: 
        (images, labels).
  """
  @staticmethod
  def load_validation_dataset(mode=None, framework='tensorflow', type=None):
    if mode is None:
      raise ValueError("Please specify a supported mode.")
    if mode not in ['only_labels',
                    'alexnet', 
                    'resnet18',
                    'efficientnetb0',
                    'mobilenet',
                    'mobilenetv2',
                    'resnet50',
                    'resnet152',
                    'vgg16',
                    'ViT-b_16p']:
      e = f"Invalid mode. {mode} not supported."
      raise ValueError(e)
    if framework not in ['tensorflow', 'torch']:
      e = f"Invalid framework. {framework} not supported."
      raise ValueError(e)

    ImageNetDataset._mode = mode
    ImageNetDataset._framework = framework
    ImageNetDataset._type = type

    if ImageNetDataset._framework == 'torch':
      # !only first N_TORCH_IMPORTED_MODELS_LOADED_IMAGES validation images and labels are loaded
      if ImageNetDataset._type is None:
        return ImageNetDataset._load_torchvision_dataset()
      elif ImageNetDataset._type == 'numpy':
        return ImageNetDataset._load_torchvision_dataset_as_numpy()
      else:
        e = f"Invalid <type> argument. {ImageNetDataset._type} not supported."
        raise ValueError(e)
      
    elif ImageNetDataset._framework == 'tensorflow':
      # !only first N_SAMPLES validation images and labels are loaded
      image_paths = []
      labels = []
      for line in open(TXT_LABELS_PATH):
        image_path, label = line.split(" ")
        image_paths.append(IMAGES_DIR_PATH+image_path)
        labels.append(int(label)) 
      tf.constant(image_paths)
      tf.constant(labels)
      dataset = tf.data.Dataset.from_tensor_slices((image_paths[:N_SAMPLES], labels[:N_SAMPLES]))
      dataset = dataset.map(ImageNetDataset._load_and_preprocess_image)
      dataset = dataset.batch(DS_BATCH_SIZE)
      dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
      images, labels = dataset.as_numpy_iterator().next()
      if ImageNetDataset._mode == 'only_labels':
        return (None, None), (None, labels)
      return (None, None) , (images, labels)
  
  """
  (public, static)
  Print the image and the corresponding label.

  Args:
    images: numpy.ndarray
      The images to be displayed.
    labels: numpy.ndarray
      The labels of the images.
    index: int
      The index of the image to be displayed.
  """
  @staticmethod
  def print_image(images, labels, index):
    image = images[index]
    label = labels[index]
    plt.imshow(image)
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()