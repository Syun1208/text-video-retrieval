from utils.translate_processing import Translation
from vit_jax import models
import time
import os
import json
import matplotlib.pyplot as plt
from langdetect import detect
import numpy as np
import faiss
import math


def time_complexity(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        print('Time inference of {}: {}'.format(args[0].mode, time.time() - start))
        return results
    return wrapper


class FaissSearch(Translation):
    def __init__(self, dict_bert_search = 'data/OCR_ASR/keyframes_id_bert.json', bin_file = 'data/OCR_ASR/faiss_beart_LIT.bin', mode = 'write', model = "lit"):
        self.mode = mode
        self.index = self.load_bin_file(bin_file)
        self.id2img_fps = self.load_json_file(dict_bert_search)
        if model == "lit":
            os.system('TF_CPP_MIN_LOG_LEVEL=0')
            self.lit_model = models.get_model("LiT-B16B")
            self.lit_var = self.lit_model.load_variables()
            self.tokenizer = self.lit_model.get_tokenizer()
            self.translate = Translation()
        else:
            pass
    def load_bin_file(self, bin_file: str):
        return faiss.read_index(bin_file)
    def load_json_file(self, json_bath: str):
        with open(json_bath, "r") as f:
            js = json.loads(f.read())
        return {int(k): v for k, v in js.items()}
    
    def show_images(self, image_paths):
        fig = plt.figure(figsize=(15, 10))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths)/columns))

        for i in range(1, columns*rows +1):
            img = plt.imread(image_paths[i - 1])
            ax = fig.add_subplot(rows, columns, i)
            ax.set_title('/'.join(image_paths[i - 1].split('/')[-3:]))

            plt.imshow(img)
            plt.axis("off")

        plt.show()

    @time_complexity
    def text_search(self, text, k):
        if detect(text) == "vi":
            text = self.translate(text)
            print("Text translation: ", text)
            tokens = self.tokenizer([text])
            _, text_features, _ = self.lit_model.apply(self.lit_var, tokens=tokens)
            scores, idx_image = self.index.search(np.array(text_features), k=k)
            idx_image = idx_image.flatten()
            infos_query = list(map(self.id2img_fps.get, list(idx_image)))
            image_paths =  [os.path.join(info['video_path'],f"{info['list_shot_id'][0]}.jpg") for info in infos_query]
            return scores, idx_image, infos_query, image_paths


def main():
    faiss_search = FaissSearch(dict_bert_search='data/OCR_ASR/keyframes_id_bert.json', bin_file='data/OCR_ASR/faiss_bert_ASR_LIT.bin', mode='search')
    text = "bộ y tế khuyến cáo không dùng methanol"
    scores, idx_image, infos_query, image_paths = faiss_search.text_search(text, k=9)
    faiss_search.show_images(image_paths)

    
if __name__ == "__main__":
    main()
