import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
import sys
from pathlib import Path
import os
from PIL import Image
import glob
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
WORK_DIR = '/'.join(WORK_DIR.split('/')[:-1])

list_model = [name for name in models.model_configs.MODEL_CONFIGS if name.startswith('LiT')]
print(list_model)

model_name = 'LiT-B16B'

lit_model = models.get_model(model_name)
# Loading the variables from cloud can take a while the first time...
lit_variables = lit_model.load_variables()
# Creating tokens from freeform text (see next section).
tokenizer = lit_model.get_tokenizer()
# Resizing images & converting value range to -1..1 (see next section).
image_preprocessing = lit_model.get_image_preprocessing()
# Preprocessing op for use in tfds pipeline (see last section).
pp = lit_model.get_pp()


@jax.jit
def embed_images(variables, images):
  zimg, _, _ = lit_model.apply(variables, images=images)
  return zimg


def main():
    # define datasets path
    npy_folder = os.path.join(WORK_DIR, 'models/lit_features')
    path = os.path.join(WORK_DIR, 'data/news')
    if not os.path.exists(npy_folder):
        os.makedirs(npy_folder)

    keyframes_paths = sorted(glob.glob(f"{path}/*/"))
    for keyframe in tqdm.tqdm(keyframes_paths, desc='Exporting LiT visual features'):
        for video in sorted(os.listdir(keyframe)):
            re_features = []
            name_video = video.split('/')[-1]
            for image_path in sorted(os.listdir(os.path.join(keyframe, video))):
                image_global_path = os.path.join(keyframe, video, image_path)
                image = image_preprocessing([Image.open(image_global_path)])
                # convert image np.array -> torch.tensor
                image_feats = embed_images(lit_variables, image)
                # normalize vector features
                # print(image_feats.shape)
                # image_feats /= np.linalg.norm(image_feats, axis=-1)
                re_features.append(image_feats)

            # create folder to save
            npy_keyframe = os.path.join(npy_folder, keyframe.split('/')[-2])
            if not os.path.exists(npy_keyframe):
                os.makedirs(npy_keyframe)
            # save vector features
            outfile = f'{npy_keyframe}/{name_video}.npy'
            np.save(outfile, re_features)

if __name__ == "__main__":
    main()