import numpy as np
import faiss
import pandas as pd
from langdetect import detect
import argparse
from PIL import Image
import numpy as np
import glob
import os
import numpy as np
import os
import pandas as pd
import glob
import torch
from sentence_transformers import SentenceTransformer



####################### ArgumentParser ##############################
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--bin_path', default='/home/hoangtv/Desktop/Nhan_CDT/text-video-retrieval/update_data/process_data/faiss_BKAI_OCR_cosine.bin', type=str, help='folder path to input images')
args = parser.parse_args()
####################################################################

torch.jit.enable_onednn_fusion(True)
torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class MyFaiss:
  def __init__(self, bin_file: str, mode = "lit"):    
    self.index = self.load_bin_file(bin_file)
    self.model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')

      
  def load_bin_file(self, bin_file: str):
    return faiss.read_index(bin_file)
  
  def text_search(self, text, k):
    text = self.model.encode(text).reshape(1,-1)
    scores, idx_image = self.index.search(np.array(text), k=k)
    idx_image = idx_image.flatten()
    # print("idx_img ",idx_image)
    return idx_image
  
def main():
    bin_path = args.bin_path
    faiss_search = MyFaiss(bin_file= bin_path)
    text = "có nhiều đối thủ tham gia như taxi điện"
    idx_image = faiss_search.text_search(text, k=30)
    print("idx image", idx_image+1)

if __name__ == '__main__':
    main()