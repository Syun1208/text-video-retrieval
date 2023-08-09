from typing import Any
import numpy as np
import glob
import time
import json
import matplotlib.pyplot as plt
import os
import math
import torch
import pandas as pd
import clip
import re
import tqdm
import sys
from pathlib import Path
from transformers import BertTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cityblock, mahalanobis
from elasticsearch import Elasticsearch, ConnectionError
from elasticsearch_dsl import Search
from elasticsearch.helpers import bulk
from langdetect import detect
from translate_processing import Translation

# Read current file path
FILE = Path(__file__).resolve()
# Read folder containing file path
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
# main work directory
WORK_DIR = os.path.dirname(ROOT)

        
ELASTIC_PASSWORD = 'v54U0ClMTS5kgiIWCFlnSdix'
CLOUD_ID = 'aic2023:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyRlOGRhNmQ3ODBmZDk0OTFiOTI2NDg3MGU2MmY4OWJiNyRiOGE4MTVjODRmNWM0MzliOTk3N2E3MGFmZWE1MTM3Zg=='
ES_ENDPOINT = f'https://aic2023.es.us-central1.gcp.cloud.es.io:9200'
INDEX_NAME = 'clip_search'
API_KEY = 'essu_ZFRob2FqSkphMEpOU2sxMGJWVlRjamhuTUVvNmFGbHNZMEZhZVZCVFNHMUhhMEV3ZFZCWmFGTkdRUT09AAAAAC7MaOo='

def time_complexity(func):
    def warp(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print('Time inference: {} second'.format(time.time() - start))
        return result
    return warp

class Similarity:
    def __init__(self) -> None:
        pass

    @staticmethod
    @time_complexity
    def cosine_similarity(vector_query, vector_db):
        return vector_query@vector_db.T / (np.linalg.norm(vector_query) * np.linalg.norm(vector_db))

    @staticmethod
    @time_complexity
    def correlation_coefficient(vector_query, vector_db):
        return np.average(np.corrcoef(vector_query, vector_db))

    @staticmethod
    @time_complexity
    def euclidean_distance(vector_query, vector_db):
        return np.linalg.norm(vector_query - vector_db)

    @staticmethod
    @time_complexity
    def manhattan(vector_query, vector_db):
        '''
        Top 1 fastest formular
        '''
        return cityblock(vector_query.flatten(), vector_db.flatten())

    # @staticmethod
    # @time_complexity
    # def mahalanobis(vector_query, vector_db):
    #     covariance_matrix = np.cov(vector_query.T, rowvar=False)
    #     return mahalanobis(vector_query.flatten(), vector_db.flatten(), np.linalg.inv(covariance_matrix).flatten())


class ElasticSearch(Translation):
    def __init__(self, folder_features, annotation, mode='plip') -> None:
        super(ElasticSearch, self).__init__(mode='googletrans')

        # Create the client instance
        self.es = Elasticsearch(
              cloud_id=CLOUD_ID,
              http_auth=("elastic", ELASTIC_PASSWORD)
          )

        self.folder_features = folder_features # folder feature path
        self.keyframes_id = self.load_json_file(annotation) # read keyframes_id.json
        # configure plip mode
        self.mode = mode
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.args_plip = parse_args()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # query dictionary in elasticsearch

        self.query = {
            "query": {
              "script_score": {
                "query" : {
                  "bool" : {
                    "filter" : {
                      "range" : {
                        "price" : {
                          "gte": 1000
                        }
                      }
                    }
                  }
                },
                "script": {
                  "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                  "params": {
                    "query_vector": []
                  }
                }
              }
            },
            "size": 10
          }


    def load_json_file(self, json_path: str):
        with open(json_path, 'r') as f:
            js = json.loads(f.read())
        return js

    def reset_indexing(self):
        s = Search(using=self.es, index=INDEX_NAME).query('match_all')  
        response = s.delete()
        print('Clear database sucessfully !')

    def create_an_index(self, VECTOR_DIMENSION = 512):
        index_settings = {
              "mappings": {
                  "properties": {
                      "vector": {
                          "type": "dense_vector",
                          "dims": VECTOR_DIMENSION  # Specify the dimension of your vector
                      },
                      "title": {
                          "type": "text"
                      },
                      "abstract": {
                          "type": "text"
                      }
                  }
              }
          }
        self.es.indices.create(index=INDEX_NAME)
        print('Create indexing sucessfully !')

    def delete_an_index(self):
        self.es.indices.delete(index=INDEX_NAME)
        print('Delete indexing sucessfully !')

    # def test_indexing(self):
    #   for i in range(1, 10):
    #       document = {
    #           'vector': np.random.randn(1,512).tolist()[0],
    #       }
    #       self.es.index(index=INDEX_NAME, document=document, id=i)

    def indexing(self):
        for values in tqdm.tqdm(self.keyframes_id.values(),desc='Indexing features to server elasticsearch'):
            image_path = values["image_path"]
            image_path = image_path.replace("Database", 'data/news')
            video_name = image_path.split('/')[-2] + '.npy'
            if video_name == 'C00_V0001.npy':
               break
            video_id = re.sub('_V\d+', '', image_path.split('/')[-2])
            batch_name = image_path.split('/')[-3].split('_')[-1]
            lip_name = f"models/{self.mode}_features/KeyFrames{video_id}_{batch_name}"
            feat_path = os.path.join(ROOT, lip_name, video_name)
            feats = np.load(feat_path)
            ids = os.listdir(re.sub('/\d+.jpg','',os.path.join(ROOT,image_path)))
            ids = sorted(ids, key=lambda x:int(x.split('.')[0]))
            id = ids.index(image_path.split('/')[-1])
            feat = feats[id]
            feat = feat.astype(np.float32).reshape(1,-1)
            # add feature to elasticsearch server
            document = {
                'vector': feat.tolist()[0],
            }
            self.es.index(index=INDEX_NAME, document=document, id=id)
            # bulk(self.es, document, index=INDEX_NAME)



    def caption_to_tokens(self, caption):
        result = self.tokenizer(caption, padding="max_length", max_length=64, truncation=True, return_tensors='pt')
        token, mask = result["input_ids"], result["attention_mask"]
        return token, mask

    def get_mode_extract(self, data, method='text'):
        '''
        Input: data = {text | image}
        Return: vector embedding
        '''

        
        if method == 'text':
            if self.mode == 'plip':
                pass
                # model = Create_PLIP_Model(self.args_plip).to(self.device)
                # token, mask = self.caption_to_tokens(data)
                # with torch.no_grad():
                #     text_feat = model.get_text_global_embedding(token,mask)
                # data_embedding[method] = text_feat.cpu().detach().numpy().astype(np.float32).flatten()
        
            elif self.mode == 'clip':
                if detect(data) == 'vi':
                    text = Translation.__call__(self, data)
                else:
                    pass # grammar detection and correction
                text = clip.tokenize([text]).to(self.device)
                model, preprocess = clip.load("ViT-B/16", device=self.device)  
                data_embedding= model.encode_text(text).cpu().detach().numpy().astype(np.float32)

        elif method == 'image':
            result = self.es.get(index=INDEX_NAME, id=data)
            data_embedding = np.array(result['_source']['vector'])


        return data_embedding
    
    def get_search_results(self):

        result = self.es.search(index=INDEX_NAME, body=self.query)

        images_id = []
        scores = []

        for hit in result['hits']['hits']:
            image_id = hit['_id']
            score = hit['_score']  # Get the similarity score
            images_id.append(image_id)
            scores.append(score)

        infos_query = list(map(self.keyframes_id.get, list(images_id)))
        image_paths = [info['image_path'] for info in infos_query]
        return images_id, scores, infos_query, image_paths


    def text_search(self, text, k):
        text_features = self.get_mode_extract(text, method='text')
        self.query['size'] = k
        self.query['query']['script_score']['script']['params']['query_vector'] = text_features.tolist()
        images_id, scores, infos_query, image_paths = self.get_search_results()
        return images_id, scores, infos_query, image_paths
    
    @time_complexity
    def video_search(self, video_id, k):
        pass

    @time_complexity
    def image_search(self, image_id, k):
        image_features = self.get_mode_extract(image_id, method='image')
        self.query['size'] = k
        self.query['query']['script_score']['script']['params']['query_vector'] = image_features.tolist()
        images_id, scores, infos_query, image_paths = self.get_search_results()
        return images_id, scores, infos_query, image_paths
    
    def show_images(self, image_paths):
        fig = plt.figure(figsize=(15, 10))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths)/columns))
        folder_save = os.path.join(WORK_DIR, 'results')
        if not os.path.exists(folder_save):
            os.makedirs(folder_save)

        for i in range(1, columns*rows +1):
            img = plt.imread(image_paths[i - 1])
            ax = fig.add_subplot(rows, columns, i)
            ax.set_title('/'.join(image_paths[i - 1].split('/')[-3:]))

            plt.imshow(img)
            plt.axis("off")
            plt.savefig(img, os.path.join(folder_save, image_paths))
        
        plt.show()
    
    def submit():
        pass

    @time_complexity
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


def main():
    #define useful path
    data_path = os.path.join(ROOT, 'data')
    model_path = os.path.join(WORK_DIR , 'models')
    keyframes_id_path = os.path.join(data_path, 'dicts/keyframes_id.json')
    folder_features = os.path.join(model_path, 'plip_features')


    # Create an object vector search
    es = ElasticSearch(folder_features, keyframes_id_path, mode='clip')
    # Set up Elasticsearch server
    try:
          # Successful response!
          es.es.info()
          print('Successful response!')
    except ConnectionError as e:
          print(e)


    # Indexing to elastic database
    es.delete_an_index()
    es.create_an_index()
    es.indexing()


    # text search: text2image, text, asr2text, ocr
    text = 'Áo đen đeo khẩu trang màu đen'
    images_id, scores, infos_query, image_paths = es.text_search(text, k=10)
    es.show_images(image_paths)


    # image search: image2text, image, asr2text, ocr
    images_id, scores, infos_query, image_paths = es.image_search(image_id=2, k=10)
    es.show_images(image_paths)


    # video search: image2text, image, asr2text, ocr


if __name__ == "__main__":
    main()
