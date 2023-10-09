from vit_jax import models
from .translate_processing import Translation
import numpy as np
import faiss
import pandas as pd
import re
import os
import sys
import time
from pathlib import Path
from langdetect import detect, DetectorFactory

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
            print('Time inference of {} and ocr: {}'.format(args[0].mode, time.time() - start))
            return results
        else:
            return func(*args, **kwargs)
    return wrapper

class OCRSearch:
    def __init__(self, info_file=os.path.join(WORK_DIR, 'data/dicts/info_ocr_loc.txt') , root_img='data/news', mode = "lit", show_time_compute=True):
        self.df_ocr = pd.read_csv(info_file, delimiter=",", header=None)
        self.translate = Translation()
        self.index = None
        self.mode = mode
        self.show_time_compute = show_time_compute
        self.root_img = root_img

        if self.mode == "lit":
            os.system('TF_CPP_MIN_LOG_LEVEL=0')
            self.lit_model = models.get_model("LiT-B16B")
            self.lit_var = self.lit_model.load_variables()
            self.tokenizer = self.lit_model.get_tokenizer()
        else:
            pass

    def process_name_img(self, name: int):
        """
        Input: 231
        Output 000231
        """
        return "0"*(6-len(str(name))) + str(name)

    # def convert_idx_to_path(self,idx_image):
    #     """
    #     Input: 204 (indx of image in txt file)
    #     Output: data/Keyframe_C00/C00_V0000/004431.jpg
    #     """
    #     path_arr = []
    #     for idx in idx_image:
    #         path = f"{self.root_img}/Keyframe{self.df_ocr[0][idx].split('_')[0]}_V{self.df_ocr[0][idx].split('_')[0][1:]}/{self.df_ocr[0][idx]}/{self.process_name_img(self.df_ocr[1][idx])}.jpg"
    #         path_arr.append(path)
    #     return path_arr

    def load_bin_file(self, bin_file=os.path.join(WORK_DIR, f'models/faiss_ocr_{mode_compute}.bin')):
        self.index = faiss.read_index(bin_file)

    @time_complexity
    def text_search(self, text, k = 9):
        if detect(text) == 'vi':
            text = self.translate(text)
        text = self.translate(text)
        print("Text translation: ", text)
        tokens = self.tokenizer([text])
        _, text_features, _ = self.lit_model.apply(self.lit_var, tokens=tokens)
        scores, idx_image = self.index.search(np.array(text_features), k=k)
        idx_image = idx_image.flatten()
        # arr_path = self.convert_idx_to_path(idx_image)

        return scores, idx_image
    
def main():
    faiss_search = OCRSearch(mode='lit')
    faiss_search.load_bin_file()
    text = "tphcm trẻ mắc bệnh tay chân miệng tăng gấp lần trong một tháng"
    scores, idx_image = faiss_search.text_search(text, k=9)
    print(idx_image)
    # Output: ['data/KeyFramesC00_V00/C00_V0000/004943.jpg', 'data/KeyFramesC00_V00/C00_V0000/004886.jpg', 'data/KeyFramesC00_V00/C00_V0000/004753.jpg', ...]

if __name__ == "__main__":
    main()