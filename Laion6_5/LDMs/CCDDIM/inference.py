import keras
import numpy as np
import yaml
import clip

from utils import get_text_encodings, plot_images
from diffuser import Diffuser

from huggingface_hub import hf_hub_download
import shutil
import os

def perform_inference(config_file):
  with open(config_file, 'r') as f:
    config = yaml.safe_load(f)
  
  #local_path = hf_hub_download(repo_id='apapiu/diffusion_model_aesthetic_keras', filename='model_64_65_epochs.h5', repo_type="model")
  #shutil.copy(local_path, os.path.join('../../models_data/', 'model_64_65_epochs.h5'))
  #autoencoder = keras.models.load_model('../../models_data/model_64_65_epochs.h5')

  autoencoder = keras.models.load_model(config['MODEL_NAME'])

  big_diffuser = Diffuser(autoencoder,
                          class_guidance=config['CLASS_GUIDANCE'],
                          diffusion_steps=config['DIFFUSION_STEPS'])

  model, _ = clip.load("ViT-B/32")
  model.cuda().eval()

  np.random.seed(config['SEED'])
  rand_image = np.random.normal(0, 1, (config['NUM_IMGS'], config['IMAGE_SIZE'], config['IMAGE_SIZE'], config['NUM_CHANNELS']))

  prompt=[config['PROMPT']]*config['NUM_IMGS']
  labels = get_text_encodings(prompt, model)

  imgs = big_diffuser.reverse_diffusion(rand_image, labels, show_img=False)
  plot_images(imgs, nrows=np.sqrt(config['NUM_IMGS']), save_name=prompt[0], size=12)

if __name__ == '__main__':
  perform_inference('inference_config.yaml')