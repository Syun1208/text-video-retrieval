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
import torch
import os
import pandas as pd
import glob
from sentence_transformers import SentenceTransformer

import sys
sys.path.append(os.path.join('/home/hoangtv/Desktop/Long/text-video-retrieval', 'utils'))

from lip_processing.blip_processing import *

####################### ArgumentParser ##############################
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--bin_path', default='/home/hoangtv/Desktop/Nhan_CDT/text-video-retrieval/update_data/lab/faiss_BKAI_OCR_cosine.bin', type=str, help='folder path to input images')
args = parser.parse_args()
####################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
model = blip_model(pretrained=model_url, image_size=384, vit='base')
model.eval()
model = model.to(device=device)

class MyFaiss:
  def __init__(self, bin_file: str, mode = "lit"):    
    self.index = self.load_bin_file(bin_file)

      
  def load_bin_file(self, bin_file: str):
    return faiss.read_index(bin_file)
  
  def text_search(self, text, k):
    with torch.no_grad():
        text_embeddings = model.get_text_features(text, device=device).squeeze(0).cpu().numpy().reshape((1,-1))
        print("Shape of text: ", text_embeddings.shape)
    scores, idx_image = self.index.search(text_embeddings, k=k)
    idx_image = idx_image.flatten()
    print("idx_img ",idx_image)
    return idx_image
  
def main():
    bin_path = args.bin_path
    faiss_search = MyFaiss(bin_file= bin_path)
    text = "Maria Van Kerkhove"
    idx_image = faiss_search.text_search(text, k=9)
    print("idx image", idx_image+1)

if __name__ == '__main__':
    main()