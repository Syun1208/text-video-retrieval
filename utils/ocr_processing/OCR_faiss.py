import numpy as np
import os
import pandas as pd
import argparse
import faiss
import tqdm



####################### ArgumentParser ##############################
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--bin_path', default='/home/hoangtv/Desktop/Nhan_CDT/text-video-retrieval/update_data/lab', type=str, help='folder path to input images')
parser.add_argument('--npy_path', default='/home/hoangtv/Desktop/Nhan_CDT/text-video-retrieval/update_data/process_data/demo_npy.npy', type=str, help='folder path to input images')

parser.add_argument('--txt_path', default='/home/hoangtv/Desktop/Nhan_CDT/text-video-retrieval/update_data/process_data/infor_search.txt', type=str, help='folder path to input images')

args = parser.parse_args()
####################################################################




def write_bin_file_ocr(bin_path: str,npy_path: str ,txt_path: str, method='cosine', feature_shape= 768): # Edit 256, 512, 768
  df_ocr = pd.read_csv(txt_path, delimiter=",", header=None)

  if method in 'L2':
    index = faiss.IndexFlatL2(feature_shape)
  elif method in 'cosine':
    index = faiss.IndexFlatIP(feature_shape)
  else:
    assert f"{method} not supported"
  
  feats = np.load(npy_path)
  print(len(feats))
  for idx in tqdm.tqdm(range(len(feats))):



 # Output ID 1, 2, 3, 4 . if len path == id reset id
    feat = feats[idx]

    feat = feat.astype(np.float32).reshape(1,-1)
    index.add(feat)
    

  faiss.write_index(index, os.path.join(bin_path, f"faiss_BKAI_OCR_{method}_hp.bin"))

  print(f'Saved {os.path.join(bin_path, f"faiss_OCR_{method}.bin")}')
  print(f"Number of Index: {idx}")

if __name__ == '__main__':
    bin_path = args.bin_path
    txt_path = args.txt_path
    npy_path = args.npy_path
    write_bin_file_ocr(bin_path, npy_path, txt_path)

