import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import sys
import cv2
import faiss
from translate_processing import Translation
import json
from typing import List
import tensorflow as tf
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import math
from pathlib import Path
import time
import numpy as np
import tqdm
from PIL import Image
import re
from vit_jax import models
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer, util

from lip_processing.blip_processing import blip_model

import clip
import torch
import pandas as pd
import shutil
import logging
from tensorflow.python.client import device_lib



# print(device_lib.list_local_devices())
# Read current file path
FILE = Path(__file__).resolve()
# Read folder containing file path
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
# main work directory
WORK_DIR = os.path.dirname(ROOT)
DATA_DIR = '/media/hoangtv/New Volume/backup/'
DetectorFactory.seed = 0
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0:2], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_memory_growth(gpus[1], True)


mode_compute = 'lit'
#define useful path
data_path = os.path.join(WORK_DIR, 'data')
folder_features = os.path.join('/media/hoangtv/New Volume/backup/Long', 'models')
# keyframes_id_path = os.path.join(data_path, 'dicts/keyframes_id.json')
keyframes_id_search_path = os.path.join(data_path, 'dicts/keyframes_id_search.json')
bin_path = os.path.join(folder_features, f'faiss_base_{mode_compute}.bin')
result_path = os.path.join(WORK_DIR, 'results')
mode_result_path = os.path.join(result_path, mode_compute)

def time_complexity(func):
    def wrapper(*args, **kwargs):
        if args[0].show_time_compute:
            start = time.time()
            results = func(*args, **kwargs)
            print('Time inference of {} and {}: {}'.format(args[0].mode, args[0].solutions, time.time() - start))
            return results
        else:
            return func(*args, **kwargs)
    return wrapper

class FaissSearch(Translation):

    def __init__(self, folder_features=folder_features, annotation=keyframes_id_search_path, mode='lit', solutions='base line', show_time_compute=True):
        super(FaissSearch, self).__init__()
        self.mode = mode
        self.solutions = solutions
        self.show_time_compute = show_time_compute
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
            self.image_preprocessing = self.lit_model.get_image_preprocessing()

        elif self.mode == 'clip':    
            self.model_clip, self.preprocess = clip.load("ViT-B/32", device=self.device) 

        elif self.mode == 'blip':
            image_size = 384
            self.model_blip = blip_model(image_size=image_size, vit='base')
            self.model_blip.eval()
            self.model_blip = self.model_blip.to(device=self.device)

        elif self.mode == 'clip-v14':
            self.model_clip_v14 = SentenceTransformer('clip-ViT-L-14', device=self.device)

        elif self.mode == 'bkai':
            self.model_bkai = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder',device=self.device)


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

        # image_paths = [self.keyframes_id[str(item)]['image_path'] for item in list(idx_image)]
        # image_paths = list(map(lambda x: x.replace("Database", 'data/news'), image_paths))
        if self.mode == 'blip':
            image_paths = [self.keyframes_id[i] for i in idx_image]
        else:
            image_paths = [self.keyframes_id[i] for i in idx_image]
        return scores, idx_image, image_paths
    


    def embed_images(self, images):
        zimg, _, _ = self.lit_model.apply(self.lit_variables, images=images)
        return zimg

    @time_complexity
    def image_search_by_id(self, image_id, k):
        query_feats = self.cpu_index.reconstruct(image_id).reshape(1,-1)
        # gpu_index = faiss.index_cpu_to_all_gpus(self.cpu_index)
        scores, idx_image = self.cpu_index.search(query_feats, k=k)
        idx_image = idx_image.flatten()

        # image_paths = [self.keyframes_id[str(item)]['image_path'] for item in list(idx_image)]
        # image_paths = list(map(lambda x: x.replace("Database", 'data/news'), image_paths))
        image_paths = [self.keyframes_id[i] for i in idx_image]
        
        return scores, idx_image, image_paths
    

    @time_complexity
    def image_search_by_path(self, image_path, k):
        image_features = self.get_mode_extract(image_path)
        image_features = np.array(image_features)
        scores, idx_image = self.cpu_index.search(image_features, k=k)

        idx_image = idx_image.flatten()

        # image_paths = [self.keyframes_id[str(item)]['image_path'] for item in list(idx_image)]
        # image_paths = list(map(lambda x: x.replace("Database", 'data/news'), image_paths))
        image_paths = [self.keyframes_id[i] for i in idx_image]
        return scores, idx_image, image_paths

    def get_mode_extract(self, data, method='text'):

        if os.path.isfile(data) and method != 'image':
            print('Set mode to image because of the image path from user !')
            method = 'image'
        
        if method == 'text':
            text = data
            if detect(data) == 'vi':
                if self.mode == 'bkai':
                    text = data
                else:
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
            elif self.mode == 'blip':
                print('Choosing mode BLIP')
                print('Translated text: ', text)
                with torch.no_grad():
                    data_embedding = self.model_blip.get_text_features(text, device=self.device ).cpu().detach().numpy().astype(np.float32)
            elif self.mode == 'clip-v14':
                print('Choosing mode CLIP v14')
                print('Translated text: ', text)
                data_embedding = self.model_clip_v14.encode(text).reshape(1, -1)

            elif self.mode == 'bkai':
                print('Choosing mode BKAI')
                print('Vietnamese text: ', text)
                data_embedding = self.model_bkai.encode(text).reshape(1, -1)

            else:
                print(f'Not found model {self.mode}')

        elif method == 'image':
            if self.mode == 'lit':
                print('Choosing mode LiT')
                image = self.image_preprocessing([Image.open(data)])
                data_embedding = self.embed_images(image)
            elif self.mode == 'clip':
                image = self.preprocess(Image.open(data)).unsqueeze(0).to(self.device)
                data_embedding = self.model_clip.encode_text(data)
                data_embedding = data_embedding.cpu().detach().numpy.astype(np.float32)
            elif self.mode == 'blip':
                image = Image.open(data)
                with torch.no_grad():
                    data_embedding = self.model_blip.get_image_features(image, device=self.device).cpu().detach().numpy().astype(np.float32)
            elif self.mode == 'clip-v14':
                data_embedding = self.model_clip_v14.encode(Image.open(data))

            else:
                print(f'Not found model {self.mode}')
        else:
                print(f'Not found method {method}')
                
        return data_embedding
    
    def indexing(self, VECTOR_DIMENSIONS, save_indexing=f'faiss_base.bin'):
        self.cpu_index = faiss.IndexFlatIP(VECTOR_DIMENSIONS)
        for i, image_path in tqdm.tqdm(enumerate(self.keyframes_id), desc=f'Indexing {self.mode} features to faiss'):
            video_name = image_path.split('/')[-2] + '.npy'
            video_id = re.sub('_V\d+', '', image_path.split('/')[-2])
            lip_name = f'{self.mode}_features_aic2023/Keyframes_{video_id}'
            feat_path = os.path.join(self.folder_features, lip_name, video_name)
            feats = np.load(feat_path)
            ids = os.listdir(re.sub('/\d+.jpg','', os.path.join(DATA_DIR, image_path)))
            ids = sorted(ids, key=lambda x: int(x.split('.')[0]))
            id = ids.index(image_path.split('/')[-1])
            feat = feats[id]
            feat = feat.astype(np.float32).reshape(1,-1)
            self.cpu_index.add(feat)
        file_save = os.path.splitext(save_indexing)[0] + f'_{self.mode}' + os.path.splitext(save_indexing)[1]
        faiss.write_index(self.cpu_index, os.path.join(WORK_DIR, 'models', file_save))
        print('Saved to: ', os.path.join(WORK_DIR, 'models', file_save))

    def load_bin_file(self, bin_file=bin_path):
        self.cpu_index = faiss.read_index(bin_file)

    def show_images(self, image_paths, save_path, method='text'):
        os.makedirs(save_path, exist_ok=True)

        fig = plt.figure(figsize=(15, 10))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths)/columns))
        

        for i in range(1, columns*rows +1):
            # img = cv2.imread(os.path.join(DATA_DIR, image_paths[i - 1]))
            # cv2.imwrite(os.path.join(save_path, image_paths[i - 1].split('/')[-1]), img)

            img = plt.imread(os.path.join(DATA_DIR, image_paths[i - 1]))
            plt.axis("off")
            plt.imshow(img)
            plt.savefig(os.path.join(save_path, image_paths[i - 1].split('/')[-1]))
            ax = fig.add_subplot(len(image_paths), 1, i)
            ax.set_title('/'.join(image_paths[i - 1].split('/')[-3:]))

            plt.imshow(img)
            plt.axis("off")
            plt.savefig(os.path.join(save_path, f'{method}_retrieval.jpg'))
        plt.show()
    
    def submit(self, text_query, k, file_name_save):
        os.makedirs(os.path.join(WORK_DIR, f'submission_{self.mode}'), exist_ok=True)
        scores, idx_image, image_paths = self.text_search(text_query, k)
        df_submit = pd.DataFrame({'videos_idx': list(map(lambda x: x.split('/')[-2], image_paths)), 'frames_idx': list(map(lambda x: os.path.splitext(x)[0].split("/")[-1], image_paths))})
        df_submit.to_csv(os.path.join(WORK_DIR, f'submission_{self.mode}/{file_name_save}.csv'), index=False, header=False)
        shutil.make_archive(os.path.join(WORK_DIR, f'submission_{self.mode}'), 'zip', os.path.join(WORK_DIR, f'submission_{self.mode}'))

    
def main():

    os.makedirs(result_path, exist_ok=True)
    os.makedirs(mode_result_path, exist_ok=True)

    # Create an object vector search
    ngpus = faiss.get_num_gpus()
    print("number of GPUs:", ngpus)
   
    clip_search = FaissSearch(mode='clip', show_time_compute=True)
    # lit_search = FaissSearch(mode=mode_compute, show_time_compute=True)
    # blip_search = FaissSearch(annotation=os.path.join(WORK_DIR, 'data/dicts/keyframes_id_all.json'), mode='blip', show_time_compute=True)
    # image_captioning_search = FaissSearch(mode=mode_compute, show_time_compute=True)
    # ocr_search = FaissSearch(mode=mode_compute, show_time_compute=True)
    # clip_v4_search = FaissSearch(mode='clip-v14', show_time_compute=True)

    # Load and save features to file.bin in faiss
    # LiT: 768 - CLIP: 512
    clip_search.indexing(VECTOR_DIMENSIONS=512)
    # lit_search.indexing(VECTOR_DIMENSIONS=768)
    # clip_v4_search.indexing(VECTOR_DIMENSIONS=768)

    # lit_search.load_bin_file()
    # clip_search.load_bin_file(os.path.join(WORK_DIR, 'models/faiss_base_clip.bin'))
    # blip_search.load_bin_file(os.path.join(WORK_DIR, 'models/faiss_base_blip.bin'))
    # image_captioning_search.load_bin_file(os.path.join(WORK_DIR, 'models/faiss_image_captioning_lit.bin'))
    # ocr_search.load_bin_file(os.path.join(WORK_DIR, 'models/faiss_ocr_lit.bin'))

    # queries_path = os.path.join(WORK_DIR, 'queries-3-21')
    # os.makedirs(queries_path, exist_ok=True)
    # os.makedirs(os.path.join(queries_path, 'blip'), exist_ok=True)
    # os.makedirs(os.path.join(queries_path, 'ic'), exist_ok=True)
    # os.makedirs(os.path.join(queries_path, 'lit'), exist_ok=True)
    # # text search: text2image, text, asr2text, ocr
    # text = 'Cảnh quay những trang giấy được in ra từ ảnh chụp của ứng dụng Messenger. Tuy nhiên nội dung đã bị che đi, chỉ biết có một ảnh chụp màn hình của một ứng dụng ngân hàng trong những trang này.'
    
    # scores, images_id, image_paths = clip_search.text_search(text, k=9)
    # df_submit = pd.DataFrame({'file_videos': list(map(lambda x: x.split('/')[-2], image_paths)), 'frames_idx': list(map(lambda x: os.path.splitext(x)[0].split("/")[-1], image_paths))})
    # df_submit.to_csv(os.path.join(os.path.join(queries_path, 'clip'), 'query-3-11.csv.csv'), index=False, header=False)
    # clip_search.show_images(image_paths, os.path.join(queries_path, 'clip'), method='text')
    # # blip_search.submit(text, 100, 'query-3-11')

    # scores, images_id, image_paths = blip_search.text_search(text, k=9)
    # df_submit = pd.DataFrame({'file_videos': list(map(lambda x: x.split('/')[-2], image_paths)), 'frames_idx': list(map(lambda x: os.path.splitext(x)[0].split("/")[-1], image_paths))})
    # df_submit.to_csv(os.path.join(os.path.join(queries_path, 'blip'), 'query-3-11.csv.csv'), index=False, header=False)
    # blip_search.show_images(image_paths, os.path.join(queries_path, 'blip'), method='text')
    # # blip_search.submit(text, 100, 'query-3-11')

    # scores, images_id, image_paths = image_captioning_search.text_search(text, k=9)
    # df_submit = pd.DataFrame({'file_videos': list(map(lambda x: x.split('/')[-2], image_paths)), 'frames_idx': list(map(lambda x: os.path.splitext(x)[0].split("/")[-1], image_paths))})
    # df_submit.to_csv(os.path.join(os.path.join(queries_path, 'ic'), 'query-3-11.csv.csv'), index=False, header=False)
    # image_captioning_search.show_images(image_paths, os.path.join(queries_path, 'ic'), method='text')
    # # blip_search.submit(text, 100, 'query-3-11')
    
    # scores, images_id, image_paths = lit_search.text_search(text, k=9)
    # df_submit = pd.DataFrame({'file_videos': list(map(lambda x: x.split('/')[-2], image_paths)), 'frames_idx': list(map(lambda x: os.path.splitext(x)[0].split("/")[-1], image_paths))})
    # df_submit.to_csv(os.path.join(os.path.join(queries_path, 'lit'), 'query-3-11.csv.csv'), index=False, header=False)
    # lit_search.show_images(image_paths, os.path.join(queries_path, 'lit'), method='text')
    # blip_search.submit(text, 100, 'query-3-11')
    # # image search: image2text, image, asr2text, ocr
    # scores, images_id, image_paths = faiss_search.image_search_by_id(image_id=2, k=30)
    # faiss_search.show_images(image_paths, mode_result_path, method='image')
    # df_images = pd.DataFrame({'images_id': list(images_id), 'scores': scores[0]})
    # df_images.to_csv(os.path.join(mode_result_path, f'image_retrieval.csv'))
    # faiss_search.submit(text, 30, 'querry-1')
    # video search: image2text, image, asr2text, ocr'
    # scores, images_id, image_paths = faiss_search.image_search_by_path(image_path=os.path.join(DATA_DIR, "data/Keyframes_L01/L01_V020/004663.jpg"), k=12)
    # faiss_search.show_images(image_paths, mode_result_path, method='image')
    # df_images = pd.DataFrame({'images_id': list(images_id), 'scores': scores[0]})
    # df_images.to_csv(os.path.join(mode_result_path, f'image_retrieval.csv'))
    # faiss_search.submit(text, 12, 'querry-1')

def competition():
    
    class Results:
        scores: List[np.ndarray]
        idx_images: List[int]
        image_paths: List[str]    
        

    folder_test = os.path.join(WORK_DIR, 'queries-p3')
    
    clip_search = FaissSearch(mode='clip', show_time_compute=True)
    lit_search = FaissSearch(mode=mode_compute, show_time_compute=True)
    blip_search = FaissSearch(annotation=os.path.join(WORK_DIR, 'data/dicts/keyframes_id_all.json'),mode='blip', show_time_compute=True)
    image_captioning_search = FaissSearch(mode=mode_compute, show_time_compute=True)
    ocr_search = FaissSearch(mode=mode_compute, show_time_compute=True)

    lit_search.load_bin_file()
    clip_search.load_bin_file(os.path.join(WORK_DIR, 'models/faiss_base_clip.bin'))
    blip_search.load_bin_file(os.path.join(WORK_DIR, 'models/faiss_base_blip.bin'))
    image_captioning_search.load_bin_file(os.path.join(WORK_DIR, 'models/faiss_image_captioning_lit.bin'))
    ocr_search.load_bin_file(os.path.join(WORK_DIR, 'models/faiss_ocr_lit.bin'))


    def sequential_process(text, k):
        list_image_paths = []
        list_scores = []
        
        ocr_results = Results()
        # asr_results = Results()
        base_results = Results()
        base_blip_results = Results()
        # image_captioning_results = Results()

        # image_captioning_results.scores, image_captioning_results.idx_images, image_captioning_results.image_paths = image_captioning.text_search(text=text, k=k)
        base_results.scores, base_results.idx_images, base_results.image_paths = lit_search.text_search(text=text, k=k)
        base_blip_results.scores, base_blip_results.idx_images, base_blip_results.image_paths = blip_search.text_search(text=text, k=k)
        ocr_results.scores, ocr_results.idx_images, ocr_results.image_paths= ocr_search.text_search(text=text, k=k)
        # asr_results.scores, asr_results.idx_images, asr_results.image_paths = asr.text_search(text=text, k=k)

        # Concatenate lists of scores and image paths    
        list_scores = base_results.scores[0].tolist() + ocr_results.scores[0].tolist() + base_blip_results.scores[0].tolist()
        list_image_paths = base_results.image_paths + ocr_results.image_paths + base_blip_results.image_paths

        return list_scores, list_image_paths


    # for file in tqdm.tqdm(os.listdir(folder_test)):
    #     with open(os.path.join(folder_test, file), 'r') as script:
    #         text = script.read()
    #     file_name = os.path.splitext(file)[0]
    #     scores, images_id, image_paths = blip_search.text_search(text, k=100)
    #     # scores, idx_image, image_paths = faiss_search.text_search(text, 100)
    #     os.makedirs(os.path.join(result_path, f'blip/{file_name}'), exist_ok=True)
    #     df_text = pd.DataFrame({'file_videos': list(map(lambda x: x.split('/')[-2], image_paths)), 'frames_idx': list(map(lambda x: os.path.splitext(x)[0].split("/")[-1], image_paths))})
    #     df_text.to_csv(os.path.join(result_path, f'blip/{file_name}/{file_name}.csv'), index=False, header=False)
    #     blip_search.show_images(image_paths, os.path.join(result_path, f'blip/{file_name}'), method='text')
    #     blip_search.submit(text, 100, file_name)


    # for file in tqdm.tqdm(os.listdir(folder_test)):
    #     with open(os.path.join(folder_test, file), 'r') as script:
    #         text = script.read()
    #     file_name = os.path.splitext(file)[0]
    #     scores, images_id, image_paths = lit_search.text_search(text, k=100)
    #     # scores, idx_image, image_paths = faiss_search.text_search(text, 100)
    #     os.makedirs(os.path.join(result_path, f'lit/{file_name}'), exist_ok=True)
    #     df_text = pd.DataFrame({'file_videos': list(map(lambda x: x.split('/')[-2], image_paths)), 'frames_idx': list(map(lambda x: os.path.splitext(x)[0].split("/")[-1], image_paths))})
    #     df_text.to_csv(os.path.join(result_path, f'lit/{file_name}/{file_name}.csv'), index=False, header=False)
    #     lit_search.show_images(image_paths, os.path.join(result_path, f'lit/{file_name}'), method='text')
    #     lit_search.submit(text, 100, file_name)
    
    # for file in tqdm.tqdm(os.listdir(folder_test)):
    #     with open(os.path.join(folder_test, file), 'r') as script:
    #         text = script.read()
    #     file_name = os.path.splitext(file)[0]
    #     scores, images_id, image_paths = clip_search.text_search(text, k=100)
    #     # scores, idx_image, image_paths = faiss_search.text_search(text, 100)
    #     os.makedirs(os.path.join(result_path, f'clip/{file_name}'), exist_ok=True)
    #     df_text = pd.DataFrame({'file_videos': list(map(lambda x: x.split('/')[-2], image_paths)), 'frames_idx': list(map(lambda x: os.path.splitext(x)[0].split("/")[-1], image_paths))})
    #     df_text.to_csv(os.path.join(result_path, f'clip/{file_name}/{file_name}.csv'), index=False, header=False)
    #     clip_search.show_images(image_paths, os.path.join(result_path, f'clip/{file_name}'), method='text')
    #     clip_search.submit(text, 100, file_name)

    for file in tqdm.tqdm(os.listdir(folder_test)):
        with open(os.path.join(folder_test, file), 'r') as script:
            text = script.read()
        file_name = os.path.splitext(file)[0]
        scores, images_id, image_paths = image_captioning_search.text_search(text, k=100)
        # scores, idx_image, image_paths = faiss_search.text_search(text, 100)
        os.makedirs(os.path.join(result_path, f'ic/{file_name}'), exist_ok=True)
        df_text = pd.DataFrame({'file_videos': list(map(lambda x: x.split('/')[-2], image_paths)), 'frames_idx': list(map(lambda x: os.path.splitext(x)[0].split("/")[-1], image_paths))})
        df_text.to_csv(os.path.join(result_path, f'ic/{file_name}/{file_name}.csv'), index=False, header=False)
        image_captioning_search.show_images(image_paths, os.path.join(result_path, f'ic/{file_name}'), method='text')
        image_captioning_search.submit(text, 100, file_name)

    for file in tqdm.tqdm(os.listdir(folder_test)):
        with open(os.path.join(folder_test, file), 'r') as script:
            text = script.read()
        file_name = os.path.splitext(file)[0]
        scores, images_id, image_paths = ocr_search.text_search(text, k=100)
        # scores, idx_image, image_paths = faiss_search.text_search(text, 100)
        os.makedirs(os.path.join(result_path, f'ocr/{file_name}'), exist_ok=True)
        df_text = pd.DataFrame({'file_videos': list(map(lambda x: x.split('/')[-2], image_paths)), 'frames_idx': list(map(lambda x: os.path.splitext(x)[0].split("/")[-1], image_paths))})
        df_text.to_csv(os.path.join(result_path, f'ocr/{file_name}/{file_name}.csv'), index=False, header=False)
        ocr_search.show_images(image_paths, os.path.join(result_path, f'ocr/{file_name}'), method='text')
        ocr_search.submit(text, 100, file_name)


if __name__ == "__main__":
    main()
    # competition()
