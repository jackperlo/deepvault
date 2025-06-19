import os
from huggingface_hub import hf_hub_download
import tarfile
import shutil

from config import REPO_ID, FILENAME, LOCAL_DIR, IMAGES_DIR_TAR, DS_DIR, IMAGES_DIR, CAPTIONS_CSV, EMBEDDINGS_CSV

def get_dataset_from_huggingface():
  os.makedirs(LOCAL_DIR, exist_ok=True)
  os.makedirs(DS_DIR, exist_ok=True)

  if os.listdir(LOCAL_DIR):
    print(f"✅ {LOCAL_DIR} already contains data. Skipping download and extraction.")
  else: 
    print(f"🚧 Downloading 115k compressed images from Laion6.5+ dataset from Hugging Face Hub into {LOCAL_DIR} ...")
    # Download the model file
    local_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    # Copy to the target directory
    shutil.copy(local_path, os.path.join(LOCAL_DIR, FILENAME))
    print(f"✅ 115k compressed images from Laion6.5+ saved to: {LOCAL_DIR}/{FILENAME}")

    print(f"\n🚧 Starting decompression from {LOCAL_DIR}/{FILENAME} into {DS_DIR} ")
    with tarfile.open(IMAGES_DIR_TAR, 'r:gz') as imgs_tar:
      imgs_tar.extractall(path=DS_DIR)
    print(f"✅ Decompression complete.")

    shutil.move(DS_DIR+'Laion6_5_plus_115k/imgs.npy', DS_DIR)
    print(f"\n✅ Images moved to {DS_DIR} ")

    shutil.move(DS_DIR+'Laion6_5_plus_115k/captions.csv', DS_DIR)
    print(f"\n✅ Captions moved to {DS_DIR} ")

    shutil.move(DS_DIR+'Laion6_5_plus_115k/embeddings.npy', DS_DIR)
    print(f"\n✅ Embeddings moved to {DS_DIR} ")

    os.rmdir(DS_DIR+'Laion6_5_plus_115k')

if __name__ == "__main__":
  get_dataset_from_huggingface()