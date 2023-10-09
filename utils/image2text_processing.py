import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
import os
from .translate_processing import Translation
from langdetect import detect, DetectorFactory
import sys
import json
import faiss
import math
import torch
import time
from pathlib import Path
from vit_jax import models


# Read current file path
FILE = Path(__file__).resolve()
# Read folder containing file path
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
# main work directory
WORK_DIR = os.path.dirname(ROOT)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
DetectorFactory.seed = 0
mode_compute = 'lit'


def time_complexity(func):
    def wrapper(*args, **kwargs):
        if args[0].show_time_compute:
          start = time.time()
          results = func(*args, **kwargs)
          print('Time inference of {} and image captioning: {}'.format(args[0].mode, time.time() - start))
          return results
        else:
          return func(*args, **kwargs)
        
    return wrapper


class ImageCaptioningSearch(Translation):
  def __init__(self, json_path=os.path.join(WORK_DIR, 'data/dicts/keyframes_id.json'), mode='lit', show_time_compute=True):
    super(ImageCaptioningSearch, self).__init__()
    self.index = None
    self.id2img_fps = self.load_json_file(json_path)
    self.mode = mode
    self.show_time_compute = show_time_compute
    self.__device = "cuda" if torch.cuda.is_available() else "cpu"
    if self.mode == 'lit':
          model_name = ['LiT-B16B', 'LiT-L16L', 'LiT-L16S', 'LiT-L16Ti']
          self.lit_model = models.get_model(model_name[0])
          self.tokenizer = self.lit_model.get_tokenizer()
          self.lit_variables = self.lit_model.load_variables()
    # elif self.mode == 'clip':    
    #     self.model_clip, preprocess = clip.load("ViT-B/16", device=self.device) 

  def load_json_file(self, json_path: str):
      with open(json_path, 'r') as f:
        js = json.loads(f.read())

      return {int(k):v for k,v in js.items()}

  def load_bin_file(self, bin_file=os.path.join(os.path.join(WORK_DIR, f'models/faiss_image_captioning_{mode_compute}.bin'))):
    self.index = faiss.read_index(bin_file)

  def show_images(self, image_paths):
    fig = plt.figure(figsize=(15, 10))
    columns = int(math.sqrt(len(image_paths)))
    rows = int(np.ceil(len(image_paths)/columns))

    for i in range(1, columns*rows +1):
      img = plt.imread(os.path.join(WORK_DIR, image_paths[i - 1]))
      ax = fig.add_subplot(rows, columns, i)
      ax.set_title('/'.join(image_paths[i - 1].split('/')[-3:]))

      plt.imshow(img)
      plt.axis("off")

    plt.show()

  @time_complexity
  def image_search(self, id_query, k):
    query_feats = self.index.reconstruct(id_query).reshape(1,-1)

    scores, idx_image = self.index.search(query_feats, k=k)
    idx_image = idx_image.flatten()

    infos_query = list(map(self.id2img_fps.get, list(idx_image)))
    image_paths = [info['image_path'] for info in infos_query]

    return scores, idx_image, infos_query, image_paths

  @time_complexity
  def text_search(self, text, k):
    if detect(text) == 'vi':
        text = Translation.__call__(self, text)
    
    if self.mode == 'lit':
        text = self.tokenizer(text)
        text_features= self.lit_model.apply(self.lit_variables, tokens=text)[1]
        text_features = np.array(text_features)
        scores, idx_image = self.index.search(text_features, k=k)
        idx_image = idx_image.flatten()
        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        image_paths = [info['image_path'] for info in infos_query]
        image_paths = list(map(lambda x: x.replace('Database', 'data/news'), image_paths))

    return scores, idx_image, image_paths
  
if __name__ == '__main__':


    cosine_faiss = ImageCaptioningSearch(mode='lit')
    cosine_faiss.load_bin_file()

    text = 'người dẫn chương trình '
    scores, image_ids, image_paths = cosine_faiss.text_search(text, k=9)
    print(scores)
    cosine_faiss.show_images(image_paths)