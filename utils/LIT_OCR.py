from vit_jax import models
from .translate_processing import Translation
import numpy as np
import faiss
import pandas as pd
import re
import os
from langdetect import detect



class Faiss_OCR():
    def __init__(self, bin_file: str, info_file: str , root_img:str, mode = "lit"):
        self.df_ocr = pd.read_csv(info_file, delimiter=",", header=None)
        self.translate = Translation()
        self.index = self.load_bin_file(bin_file)
        self.root_img = root_img

        if mode == "lit":
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

    def convert_idx_to_path(self,idx_image):
        """
        Input: 204 (indx of image in txt file)
        Output: data/Keyframe_C00/C00_V0000/004431.jpg
        """
        path_arr = []
        for idx in idx_image:
            path = f"{self.root_img}/{self.df_ocr[0][idx]}/{self.process_name_img(self.df_ocr[1][idx])}.jpg"
            path_arr.append(path)
        return path_arr

    def load_bin_file(self, bin_file):
        return faiss.read_index(bin_file)
    
    def text_search(self, text, k = 9):
        if detect(text) == 'vi':
            text = self.translate(text)
        text = self.translate(text)
        print("Text translation: ", text)
        tokens = self.tokenizer([text])
        _, text_features, _ = self.lit_model.apply(self.lit_var, tokens=tokens)
        scores, idx_image = self.index.search(np.array(text_features), k=k)
        idx_image = idx_image.flatten()
        arr_path = self.convert_idx_to_path(idx_image)

        return arr_path
    
def main():
    faiss_search = Faiss_OCR(bin_file='data/models/faiss_LIT_OCR_cosine.bin', info_file = "data/OCR_ASR/info_ocr_loc.txt", root_img = "data/KeyFramesC00_V00")
    text = "tphcm trẻ mắc bệnh tay chân miệng tăng gấp lần trong một tháng"
    idx_image = faiss_search.text_search(text, k=9)
    print(idx_image)
    # Output: ['data/KeyFramesC00_V00/C00_V0000/004943.jpg', 'data/KeyFramesC00_V00/C00_V0000/004886.jpg', 'data/KeyFramesC00_V00/C00_V0000/004753.jpg', ...]

if __name__ == "__main__":
    main()


