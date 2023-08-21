from .translate_processing import Translation
from vit_jax import models
import time
import os
import json
import matplotlib.pyplot as plt
from langdetect import detect, DetectorFactory
import numpy as np
import faiss
import math
import sys
from pathlib import Path

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

def time_complexity(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        print('Time inference of {} and asr: {}'.format(args[0].mode, time.time() - start))
        return results
    return wrapper


class ASRSearch(Translation):
    def __init__(self, dict_bert_search = os.path.join(WORK_DIR, 'data/dicts/keyframes_id_bert.json'), bin_file = os.path.join(WORK_DIR, 'models/faiss_asr.bin'), mode = "lit"):
        self.index = None
        self.mode = mode
        self.id2img_fps = self.load_json_file(dict_bert_search)
        if self.mode == "lit":
            os.system('TF_CPP_MIN_LOG_LEVEL=0')
            self.lit_model = models.get_model("LiT-B16B")
            self.lit_var = self.lit_model.load_variables()
            self.tokenizer = self.lit_model.get_tokenizer()
            self.translate = Translation()
        else:
            pass
    def load_bin_file(self, bin_file=os.path.join(WORK_DIR, 'models/faiss_asr.bin')):
        self.index = faiss.read_index(bin_file)
    
    def load_json_file(self, json_bath: str):
        with open(json_bath, "r") as f:
            js = json.loads(f.read())
        return {int(k): v for k, v in js.items()}
    
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
    def text_search(self, text, k):
        if detect(text) == "vi":
            text = self.translate(text)
        if self.mode == 'lit':
            print("Text translation: ", text)
            tokens = self.tokenizer([text])
            _, text_features, _ = self.lit_model.apply(self.lit_var, tokens=tokens)
            scores, idx_image = self.index.search(np.array(text_features), k=k)
            idx_image = idx_image.flatten()
            infos_query = list(map(self.id2img_fps.get, list(idx_image)))
            image_paths =  [os.path.join(info['video_path'],f"{info['list_shot_id'][0]}.jpg") for info in infos_query]
            image_paths = [img_path.replace('Database', 'data/news') for img_path in image_paths]
        return scores, idx_image, image_paths


def main():
    faiss_search = ASRSearch(mode='lit')
    text = "bộ y tế khuyến cáo không dùng methanol"
    faiss_search.load_bin_file()
    scores, idx_image,image_paths = faiss_search.text_search(text, k=9)
    faiss_search.show_images(image_paths)

    
if __name__ == "__main__":
    main()