import numpy as np
import glob
import os
import pandas as pd
import glob
import json
import argparse
import tqdm
from utils.lip_processing import blip_processing
import torch
import time
from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer




######################################################################
# input: # địa chỉ các file *.txt
         # địa chỉ lưu file npy   

# output: các npy dc lưu vào file

######################################################################

torch.jit.enable_onednn_fusion(True)
torch.backends.cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')


####################### ArgumentParser ##############################
parser = argparse.ArgumentParser(description='Encode npy with blip')
parser.add_argument('--txt_path', default='/home/hoangtv/Desktop/Nhan_CDT/text-video-retrieval/update_data/lab', type=str, help='folder path to input images')
parser.add_argument('--npy_path', default='/home/hoangtv/Desktop/Nhan_CDT/text-video-retrieval/update_data/lab', type=str, help='folder path to input images')
args = parser.parse_args()
####################################################################


def main():
    paths = sorted(glob.glob(txt_paths+"/*.txt"))
    print(paths)

    for txt_path in paths:
        df_ocr = pd.read_csv(txt_path, delimiter=",", header=None)
        
        index = 0
        os.makedirs(f"{npy_path}", exist_ok = True)
        re_feats = []
        for i in tqdm.tqdm(range(len(df_ocr[2]))):
            text = df_ocr[2][index][:5000] # anhasd asada s

            # print(f"Idx: {index}")
            embeddings = model.encode(text).reshape(1,-1)


            # print("Shape of text_embeddings ", text_embeddings.shape) # (1, 256)
            index += 1
            re_feats.append(embeddings)
        outfile = f'{npy_path}/{txt_path.split("/")[-1].replace(".txt","")}.npy' # Edit
        np.save(outfile, re_feats)
        print(f"Save {outfile}")




if __name__ == '__main__':
    txt_paths = args.txt_path
    npy_path = args.npy_path
    main()