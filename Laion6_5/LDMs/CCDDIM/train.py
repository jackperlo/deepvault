import os
import yaml
import numpy as np
import pandas as pd

from keras.utils import plot_model
from matplotlib import pyplot as plt

from denoiser import get_network
from utils import batch_generator, plot_images, preprocess
from diffuser import Diffuser

class Trainer:
  def __init__(self, config_file):
    # Load configuration from YAML file and set attributes
    with open(config_file, 'r') as f:
      config = yaml.safe_load(f)
    for key, value in config.items():
      setattr(self, key, value)
    self._setup_paths_and_dirs()

  def _setup_paths_and_dirs(self):
    self.widths = [c * self.channels for c in self.channel_multiplier]
    self.captions_path = os.path.join(self.ds_data_dir, "captions.csv")
    self.imgs_path = os.path.join(self.ds_data_dir, "imgs.npy")
    self.embedding_path = os.path.join(self.ds_data_dir, "embeddings.npy")
    
    self.home_dir = self.MODEL_NAME
    if not os.path.exists(self.home_dir):
      os.makedirs(self.home_dir, exist_ok=True)

    self.model_path = os.path.join(self.home_dir, self.MODEL_NAME + ".keras")

  def preprocess_data(self, train_data, train_label_embeddings):
    print(train_data.shape)
    self.train_data = train_data
    self.train_label_embeddings = train_label_embeddings
    self.image_size = train_data.shape[1]
    self.num_channels = train_data.shape[-1]
    self.row = int(np.sqrt(self.num_imgs))
    self.labels = self._get_labels(train_label_embeddings)

  def _get_labels(self, train_label_embeddings):
    if self.precomputed_embedding:
      return train_label_embeddings[:self.num_imgs]
    else:
      row_labels = np.array([[i] * self.row for i in np.arange(self.row)]).flatten()[:, None] 
      return row_labels + 1

  def initialize_model(self):
    self.autoencoder = get_network(self.image_size,
                                    self.widths,
                                    self.block_depth,
                                    num_classes=self.num_classes,
                                    attention_levels=self.attention_levels,
                                    emb_size=self.emb_size,
                                    num_channels=self.num_channels,
                                    precomputed_embedding=self.precomputed_embedding)

    self.autoencoder.compile(optimizer="adam", loss="mae")
    print(self.autoencoder.summary())

  def data_checks(self, train_data):
    print("Number of parameters is {0}".format(self.autoencoder.count_params()))
    pd.Series(train_data[:1000].ravel()).hist(bins=80)
    plt.show()
    print("Original Images:")
    plot_images(preprocess(train_data[:self.num_imgs]), nrows=int(np.sqrt(self.num_imgs)))
    plot_model(self.autoencoder, to_file=os.path.join(self.home_dir, 'model_plot.png'),
                show_shapes=True, show_layer_names=True)
    print("Generating Images below:")

  def train(self):
    np.random.seed(0xdeadbeef)
    self.rand_image = np.random.normal(0, 1, (self.num_imgs, self.image_size, self.image_size, self.num_channels))

    self.diffuser = Diffuser(self.autoencoder,
                              class_guidance=self.class_guidance,
                              diffusion_steps=35)

    if self.train_model:
      train_generator = batch_generator(self.autoencoder,
                                        self.model_path,
                                        self.train_data,
                                        self.train_label_embeddings,
                                        self.epochs,
                                        self.batch_size,
                                        self.rand_image,
                                        self.labels,
                                        self.home_dir,
                                        self.diffuser)

      self.autoencoder.optimizer.learning_rate.assign(self.learning_rate)

      self.eval_nums = self.autoencoder.fit(
        x=train_generator,
        epochs=self.epochs
      )

def get_train_data(captions_path, imgs_path, embedding_path):
  _captions = pd.read_csv(captions_path)
  train_data, train_label_embeddings = np.load(imgs_path), np.load(embedding_path)

  return train_data, train_label_embeddings

if __name__=='__main__':
  trainer = Trainer('config.yaml')

  train_data, train_label_embeddings = get_train_data(trainer.captions_path, 
                                                      trainer.imgs_path, 
                                                      trainer.embedding_path)
  trainer.preprocess_data(train_data, train_label_embeddings)
  trainer.initialize_model()
  trainer.data_checks(train_data)
  trainer.train()