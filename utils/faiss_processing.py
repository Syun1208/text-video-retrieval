import faiss
from .translate_processing import Translation
import json
import matplotlib.pyplot as plt
import math
import os
import sys
from pathlib import Path
import time
import numpy as np
import tqdm
import re
from vit_jax import models
from langdetect import detect, DetectorFactory
import clip
import torch
import pandas as pd

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
#define useful path
data_path = os.path.join(WORK_DIR, 'data')
folder_features = os.path.join(WORK_DIR, 'models')
keyframes_id_path = os.path.join(data_path, 'dicts/keyframes_id.json')
bin_path = os.path.join(folder_features, f'faiss_{mode_compute}.bin')
result_path = os.path.join(WORK_DIR, 'results')
mode_result_path = os.path.join(result_path, mode_compute)

def time_complexity(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        print('Time inference of {}: {}'.format(args[0].mode, time.time() - start))
        return results
    return wrapper

class FaissSearch(Translation):

    def __init__(self, folder_features=folder_features, annotation=keyframes_id_path, mode='lit'):
        super(FaissSearch, self).__init__()
        self.mode = mode
        self.cpu_index = None
        # self.gpu_index = None
        self.folder_features = folder_features # folder feature path
        self.keyframes_id = self.load_json_file(annotation) # read keyframes_id.json
        self.query = {
            'encoder': [],
            'k': 1
        }   
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.mode == 'lit':
            model_name = ['LiT-B16B', 'LiT-L16L', 'LiT-L16S', 'LiT-L16Ti']
            self.lit_model = models.get_model(model_name[0])
            self.tokenizer = self.lit_model.get_tokenizer()
            self.lit_variables = self.lit_model.load_variables()
        elif self.mode == 'clip':    
            self.model_clip, preprocess = clip.load("ViT-B/16", device=self.device) 
   


    def load_json_file(self, json_path: str):
        with open(json_path, 'r') as f:
            js = json.loads(f.read())
        return js
    

    @time_complexity
    def text_search(self, text, k):
        text_features = self.get_mode_extract(text, method='text')
        text_features = np.array(text_features)  # Reshape to 2D array
        # gpu_index = faiss.index_cpu_to_all_gpus(self.cpu_index)
        scores, idx_image = self.cpu_index.search(text_features, k=k)

        idx_image = idx_image.flatten()

        image_paths = [self.keyframes_id[str(item)]['image_path'] for item in list(idx_image)]
        image_paths = list(map(lambda x: x.replace("Database", 'data/news'), image_paths))

        return scores, idx_image, image_paths
    

    @time_complexity
    def image_search(self, image_id, k):
        query_feats = self.cpu_index.reconstruct(image_id).reshape(1,-1)
        # gpu_index = faiss.index_cpu_to_all_gpus(self.cpu_index)
        scores, idx_image = self.cpu_index.search(query_feats, k=k)
        idx_image = idx_image.flatten()

        image_paths = [self.keyframes_id[str(item)]['image_path'] for item in list(idx_image)]
        image_paths = list(map(lambda x: x.replace("Database", 'data/news'), image_paths))

        
        return scores, idx_image, image_paths

    def video_search(self, video, k):
        pass

    def get_mode_extract(self, data, method='text'):

        if method == 'text':
            if detect(data) == 'vi':
                text = Translation.__call__(self, data)
            if self.mode == 'lit':
                print('Choosing mode LiT')
                print('Translated text: ', text)
                tokens = self.tokenizer([text])
                _, data_embedding, _ = self.lit_model.apply(self.lit_variables, tokens=tokens)
            elif self.mode == 'clip':
                print('Choosing mode CLIP')
                print('Translated text: ', text)
                text = clip.tokenize([text]).to(self.device)
                data_embedding = self.model_clip.encode_text(text)
                data_embedding = data_embedding.cpu().detach().numpy().astype(np.float32)
            else:
                print(f'Not found model {self.mode}')


        elif method == 'image':
            pass

        return data_embedding
    

    def indexing(self, VECTOR_DIMENSIONS):
        self.cpu_index = faiss.IndexFlatIP(VECTOR_DIMENSIONS)
        # gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
        #     self.cpu_index
        # )
        for values in tqdm.tqdm(self.keyframes_id.values(),desc=f'Indexing {self.mode} features to faiss'):
            image_path = values["image_path"]
            image_path = image_path.replace("Database", 'data/news')
            video_name = image_path.split('/')[-2] + '.npy'
            if video_name == 'C00_V0002.npy':
               break
            video_id = re.sub('_V\d+', '', image_path.split('/')[-2])
            batch_name = image_path.split('/')[-3].split('_')[-1]
            lip_name = f"{self.mode}_features/KeyFrames{video_id}_{batch_name}"
            feat_path = os.path.join(self.folder_features, lip_name, video_name)
            feats = np.load(feat_path)
            ids = os.listdir(re.sub('/\d+.jpg','',os.path.join(WORK_DIR, image_path)))
            ids = sorted(ids, key=lambda x:int(x.split('.')[0]))
            id = ids.index(image_path.split('/')[-1])
            feat = feats[id]
            feat = feat.astype(np.float32).reshape(1,-1)
            # add feature to faiss
            self.cpu_index.add(feat)
        # self.cpu_index = faiss.index_gpu_to_cpu(gpu_index)
        faiss.write_index(self.cpu_index, os.path.join(WORK_DIR, 'models', f'faiss_{self.mode}.bin'))
        print('Saved to: ', os.path.join(WORK_DIR, 'models', f'faiss_{self.mode}.bin'))

    def load_bin_file(self, bin_file=bin_path):
        self.cpu_index = faiss.read_index(bin_file)

    def show_images(self, image_paths, save_path, method='text'):
        fig = plt.figure(figsize=(15, 10))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths)/columns))
        

        for i in range(1, columns*rows +1):
            img = plt.imread(os.path.join(WORK_DIR, image_paths[i - 1]))
            ax = fig.add_subplot(rows, columns, i)
            ax.set_title('/'.join(image_paths[i - 1].split('/')[-3:]))

            plt.imshow(img)
            plt.axis("off")
            plt.savefig(os.path.join(save_path, f'{method}_retrieval.jpg'))
        plt.show()



def main():

    if not os.path.exists(os.path.join(result_path)):
        os.makedirs(result_path)

    if not os.path.exists(os.path.join(mode_result_path)):
        os.makedirs(mode_result_path)

    # Create an object vector search
    faiss_search = FaissSearch(folder_features, keyframes_id_path, mode=mode_compute)

    ngpus = faiss.get_num_gpus()
    print("number of GPUs:", ngpus)
    # Load and save features to file.bin in faiss
    # LiT: 768 - CLIP: 512
    faiss_search.indexing(VECTOR_DIMENSIONS=768)
    faiss_search.load_bin_file()

    # text search: text2image, text, asr2text, ocr
    text = 'Khung cảnh trong một sự kiện sản xuất khí cầu tại venezuela. \
        Có 3 khí cầu to lớn đang chuẩn bị được bay lên bầu trời. \
        Chuyển cảnh cuộc phổng vấn một nhóm người đang chuẩn bị trải nghiệm với một khí cầu. \
        Đại diện cho nhóm là người đàn ông đeo kính khoác một chiếc áo đỏ đen trả lời phỏng vấn về sự kiện này.'
    scores, images_id, image_paths = faiss_search.text_search(text, k=9)
    df_text = pd.DataFrame({'images_id': list(images_id), 'scores': scores[0]})
    df_text.to_csv(os.path.join(mode_result_path, f'text_retrieval.csv'))
    faiss_search.show_images(image_paths, mode_result_path, method='text')
    
    # image search: image2text, image, asr2text, ocr
    scores, images_id, image_paths = faiss_search.image_search(image_id=2, k=9)
    faiss_search.show_images(image_paths, mode_result_path, method='image')
    df_images = pd.DataFrame({'images_id': list(images_id), 'scores': scores[0]})
    df_images.to_csv(os.path.join(mode_result_path, f'image_retrieval.csv'))
    # video search: image2text, image, asr2text, ocr


if __name__ == "__main__":
    main()