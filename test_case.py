from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import time
import uvicorn
import numpy as np
import socket
import heapq
import asyncio
import base64
from concurrent.futures import ProcessPoolExecutor
import sorting
from threading import Thread
from multiprocessing import Process, Pool
import threading
from collections import Counter
from fastapi.responses import JSONResponse, RedirectResponse, Response, ORJSONResponse
from fastapi.encoders import jsonable_encoder
import sys
from pathlib import Path
import os
import pandas as pd
# from utils.asr_processing import ASRSearch

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

app = FastAPI()
HOST = socket.gethostbyname(socket.gethostname())
# Read current file path
FILE = Path(__file__).resolve()
# Read folder containing file path
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
# main work directory
WORK_DIR = os.path.dirname(ROOT)
sys.path.append(os.path.join(ROOT, 'utils'))

from utils.faiss_processing import FaissSearch

base = FaissSearch(mode='lit')
base_blip = FaissSearch(mode='blip')
ocr = FaissSearch(mode='lit', solutions='ocr')

base.load_bin_file()
ocr.load_bin_file(os.path.join(ROOT, 'models/faiss_ocr_lit.bin'))
base_blip.load_bin_file(os.path.join(ROOT, 'models/faiss_base_blip.bin'))

class UserRequest(BaseModel):
    image_paths: List[str]
    scores: List[float]

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs, Verbose)
        self._return = None
    def run(self):
        if self._Thread__target is not None:
            self._return = self._Thread__target(*self._Thread__args,
                                                **self._Thread__kwargs)
    def join(self):
        Thread.join(self)
        return self._return
    
class ProcessTestReturnValue(Process):
    def __init__(self, target=None, args=None):
        super(ProcessTestReturnValue, self).__init__()
        self.target = target
        self.args = args
        self.scores = None
        self.image_paths = None
        self.text = None
        self.k = None

    def run(self):
        # Call the target function and get the results
        scores, image_paths, text, k = self.target(self.args[0], self.args[1])
        
        # Assign the results to the instance attributes
        self.scores = scores
        self.image_paths = image_paths
        self.text = text
        self.k = k

scores_submit = []
image_paths_submit = []

@app.get("/test")
async def test():
    results = {'scores': [0.53, 0.53, 0.46, 0.89, 0.98], 'image_paths': ['afasf', 'afasfa', 'afdaf', 'asfasf', 'afaf']}
    scores_submit = results['scores']
    image_paths_submit = results['image_paths']
    print(scores_submit)
    print(image_paths_submit)
    return JSONResponse(content=results, status_code=200)

@app.get('/submit', response_class=FileResponse)
async def submit():
    file_response = os.path.join(ROOT, 'test_submit.csv')
    print(scores_submit)
    print(image_paths_submit)
    df_submit = pd.DataFrame({'scores_submits': scores_submit, 'image_paths_submit': image_paths_submit})
    df_submit.to_csv(file_response)
    response = FileResponse(file_response, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=downloaded_file.csv"
    return response
async def test1(text, k):
    for i in range(1000):
        a = 1
    scores = [0.53, 0.53, 0.46, 0.89, 0.98]
    image_paths = ['afasf', 'afasfa', 'afdaf', 'asfasf', 'afaf']
    return scores, image_paths, text, k

async def test2(text, k):
    for i in range(1000):
        a = 1
    scores = [0.53, 0.53, 0.46, 0.89, 0.98]
    image_paths = ['afasf', 'afasfa', 'afdaf', 'asfasf', 'afaf']
    return scores, image_paths, text, k
@app.get('/video_path_response')
async def video_path_response():
    video_path = os.path.join(ROOT, 'video/news/test.mp4')
    print(video_path)
    return FileResponse(video_path)

@app.get('/video_streaming')
async def video_streaming():
    video_path = os.path.join(ROOT, 'video/news/test.mp4')
    def interfile():
        with open(video_path, 'rb') as file:
            yield from file
    return StreamingResponse(interfile(), media_type='video/mp4')

@app.get('/use_async')
async def use_async():
    start = time.time()
    funcs = [test1('hello', 10), test2('hi', 15)]
    a, b = await asyncio.gather(*funcs)
    time_response = time.time() - start
    response = {
        'a': a,
        'b': b,
        'time_response': time_response
    }
    return JSONResponse(content=jsonable_encoder(response))

@app.get('/sequential')
async def sequential():
    start = time.time()
    a = test1('hello', 10)
    b = test2('hi', 15)
    time_response = time.time() - start
    response = {
        'a': a,
        'b': b,
        'time_response': time_response
    }
    return JSONResponse(content=jsonable_encoder(response))

@app.get('/non_async')
async def non_async():
    start = time.time()
    a = await test1('hello', 10)
    b = await test2('hi', 15)
    time_response = time.time() - start
    response = {
        'a': a,
        'b': b,
        'time_response': time_response
    }
    return JSONResponse(content=jsonable_encoder(response))

def test_findind_topk_x3():
    k = 10
    list_scores = [
        0.7494096159934998,
        0.7494096159934998,
        0.7494096159934998,
        0.7480238080024719,
        0.7480238080024719,
        0.7480238080024719,
        0.7471744418144226,
        0.7471744418144226,
        0.7435529828071594,
        0.7426625490188599,
        0.7411181330680847,
        0.7410913705825806,
        0.7410414218902588,
        0.7381133437156677,
        0.7376676797866821
    ]
    list_image_paths = [
        "data/news/KeyFramesC00_V00/C00_V0001/001831.jpg",
        "data/news/KeyFramesC00_V00/C00_V0001/001831.jpg",
        "data/news/KeyFramesC00_V00/C00_V0001/001831.jpg",
        "data/news/KeyframeC00_V00/C00_V0000/002138.jpg",
        "data/news/KeyframeC00_V00/C00_V0000/002138.jpg",
        "data/news/KeyframeC00_V00/C00_V0000/002138.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/023979.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/023979.jpg",
        "data/news/KeyFramesC00_V00/C00_V0001/016594.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/023646.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/002673.jpg",
        "data/news/KeyFramesC00_V00/C00_V0001/000676.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/027318.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/023656.jpg",
        "data/news/KeyframeC00_V00/C00_V0000/007956.jpg"
    ]
    start = time.time()
    # Remove duplicated image paths
    list_image_paths_nonduplicate = list(set(list_image_paths))
    list_idx_image_paths = [list_image_paths.index(string) for string in list_image_paths_nonduplicate]
    list_scores = [list_scores[idx] for idx in list_idx_image_paths]
    # Sort concatenated lists
    list_scores = sorted(list_scores)
    idx_sorted_scores = [i for i, x in sorted(enumerate(list_scores), key=lambda x: x[1])]
    list_image_paths = [list_image_paths[i] for i in idx_sorted_scores]
    # Find top K maximum of the sorted lists
    max_scores, max_image_paths = list_scores[:k], list_image_paths[:k]
    print('Time finding Top K x 3: ', time.time() - start)
    # list test cases
    scores_test =  [
        0.7494096159934998,
        0.7480238080024719,
        0.7471744418144226,
        0.7435529828071594,
        0.7426625490188599,
        0.7411181330680847,
        0.7410913705825806,
        0.7410414218902588,
        0.7381133437156677,
        0.7376676797866821
    ]
    image_path_test = [
        "data/news/KeyFramesC00_V00/C00_V0001/001831.jpg",
        "data/news/KeyframeC00_V00/C00_V0000/002138.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/023979.jpg",
        "data/news/KeyFramesC00_V00/C00_V0001/016594.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/023646.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/002673.jpg",
        "data/news/KeyFramesC00_V00/C00_V0001/000676.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/027318.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/023656.jpg",
        "data/news/KeyframeC00_V00/C00_V0000/007956.jpg"
    ]
    print(max_scores, max_image_paths)
    print(max_scores == sorted(scores_test),  max_image_paths == image_path_test)


def most_frequent(list_items):
    count = 0
    _item = list_items[0]
    for item in list_items:
        count_frequent = list_items.count(item)
        if count_frequent > count:
            count = count_frequent
            _item = item


    return _item

def most_common_indices(lst, num_most_common=1):
    counter = Counter(lst)
    most_common = counter.most_common(num_most_common)  # Returns a list of (item, count) pairs
    most_common_items = [item for item, count in most_common]
    
    n_common_items = [lst.count(item) for item in most_common_items]
    print(n_common_items)

    return most_common_items, n_common_items

def sort_image_paths_by_frequency_and_score(list_image_paths, list_scores):
    counter = Counter(list_image_paths)
    sorted_paths = sorted(counter.keys(), key=lambda path: (-counter[path], -list_scores[list_image_paths.index(path)]))
    sorted_scores = [list_scores[list_image_paths.index(sorted_path)] for sorted_path in sorted_paths]
    return sorted_scores, sorted_paths

def find_top_k_image_paths(list_scores, list_image_paths, k):
    combined_data = zip(list_scores, list_image_paths)
    sorted_combined_data = sorted(combined_data, reverse=True)  # Sort by scores in descending order
    
    top_k_data = heapq.nlargest(k, sorted_combined_data)
    top_k_image_paths = [path for score, path in top_k_data]
    top_k_image_paths = list(set(top_k_image_paths))

    if len(top_k_image_paths) < k:
        additional_paths = [(image_path, score) for score, image_path in sorted_combined_data if image_path not in top_k_image_paths]
        additional_paths = heapq.nlargest(k - len(top_k_image_paths), additional_paths)
        additional_paths = [path for path, score in additional_paths]
        additional_paths = additional_paths + top_k_image_paths
    
    additional_paths = list(set(additional_paths))

    return additional_paths


def voting_algorithm(list_scores, list_image_paths, k):
    element_counts = Counter(list_image_paths)

    top_list_image_paths = list(set([element for element, count in element_counts.most_common(k)]))
    top_list_scores = [list_scores[list_image_paths.index(image_path)] for image_path in top_list_image_paths]

    
    return top_list_scores, top_list_image_paths

    


def test_voting():
    k = 10
    list_scores = [
        0.7494096159934998,
        0.7494096159934998,
        0.7494096159934998,
        0.7480238080024719,
        0.7480238080024719,
        0.7480238080024719,
        0.7471744418144226,
        0.7471744418144226,
        0.7435529828071594,
        0.7426625490188599,
        0.7411181330680847,
        0.7410913705825806,
        0.7410414218902588,
        0.7381133437156677,
        0.7376676797866821,
        0.7376676797866821,
        0.7376676797866821
    ]
    list_image_paths = [
        "data/news/KeyFramesC00_V00/C00_V0001/001831.jpg",
        "data/news/KeyFramesC00_V00/C00_V0001/001831.jpg",
        "data/news/KeyFramesC00_V00/C00_V0001/001831.jpg",
        "data/news/KeyframeC00_V00/C00_V0000/002138.jpg",
        "data/news/KeyframeC00_V00/C00_V0000/002138.jpg",
        "data/news/KeyframeC00_V00/C00_V0000/002138.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/023979.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/023979.jpg",
        "data/news/KeyFramesC00_V00/C00_V0001/016594.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/023646.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/002673.jpg",
        "data/news/KeyFramesC00_V00/C00_V0001/000676.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/027318.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/023656.jpg",
        "data/news/KeyframeC00_V00/C00_V0000/007956.jpg",
        "data/news/KeyframeC00_V00/C00_V0000/007956.jpg",
        "data/news/KeyframeC00_V00/C00_V0000/007956.jpg"
    ]
    start = time.time()
    # top_k_image_paths = find_top_k_image_paths(list_scores, list_image_paths, k)
    # print(f"Top {len(top_k_image_paths)} image path(s):", top_k_image_paths)
    # sorted_scores, sorted_paths = sort_image_paths_by_frequency_and_score(list_image_paths, list_scores)
    sorted_scores, sorted_paths = voting_algorithm(list_scores, list_image_paths, k)
    print(sorted_paths)
    print(sorted_scores)
    print('Time voting: ', time.time() - start)
    # list test cases
    scores_test =  [
        0.7494096159934998,
        0.7480238080024719,
        0.7471744418144226,
        0.7435529828071594,
        0.7426625490188599,
        0.7411181330680847,
        0.7410913705825806,
        0.7410414218902588,
        0.7381133437156677,
        0.7376676797866821
    ]
    image_path_test = [
        "data/news/KeyFramesC00_V00/C00_V0001/001831.jpg",
        "data/news/KeyframeC00_V00/C00_V0000/002138.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/023979.jpg",
        "data/news/KeyFramesC00_V00/C00_V0001/016594.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/023646.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/002673.jpg",
        "data/news/KeyFramesC00_V00/C00_V0001/000676.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/027318.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/023656.jpg",
        "data/news/KeyframeC00_V00/C00_V0000/007956.jpg"
    ]
    print(len(sorted_paths) == len(image_path_test))
    print(len(sorted_scores) == len(scores_test))
    # print(max_scores, max_image_paths)
    # print(max_scores == sorted(scores_test),  max_image_paths == image_path_test)

def test_pop():
    start = time.time()
    scores, image_paths, text, k = test1('hello world', 3)
    print(scores, image_paths, text, k)
    scores, image_paths, text, k = test1('hi', 4)
    print(scores, image_paths, text, k)
    print('Time POP: ', time.time() - start)


def test_multi_process():
    start = time.time()

    with ProcessPoolExecutor() as executor:
        future1 = executor.submit(test1, 'hello world', 3)
        future2 = executor.submit(test2, 'hi', 4)

        scores1, image_paths1, text1, k1 = future1.result()
        scores2, image_paths2, text2, k2 = future2.result()

        print(scores1, image_paths1, text1, k1)
        print(scores2, image_paths2, text2, k2)

    print('Time: ', time.time() - start)

def video_response_processing(image_paths):
    video_results = []
    for image_path in image_paths:
        id_image_path = int(os.path.splitext(image_path.split('/')[-1])[0])
        video_database = image_path.replace('data', 'video')
        video_paths = os.listdir(os.path.join(WORK_DIR, '/'.join(video_database.split('/')[:-1])))
        video_result = [video for video in video_paths if id_image_path in np.arange(os.path.splitext(video.split('/')[-1])[0].split('_')[0], os.path.splitext(video.split('/')[-1])[0].split('_')[1])]
        video_results.extend(video_result)
    
    video_results = list(set(video_results))
    return video_results

def check_id_image_in_video(image_path, video_path):
    id_video = os.path.splitext(video_path.split('/')[-1])[0].split('_')
    id_video_start = id_video.split[-3]
    id_video_stop = id_video.split[-1]
    id_image_path = int(os.path.splitext(image_path.split('/')[-1])[0])
    if id_image_path in np.arange(id_video_start, id_video_stop):
        return True
    return False

class Results:
    scores: List[np.ndarray]
    idx_images: List[int]
    image_paths: List[str]    

def sequential_process(text, k):
    list_image_paths = []
    list_scores = []
    
    ocr_results = Results()
    base_results = Results()

    base_results.scores, base_results.idx_images, base_results.image_paths = base.text_search(text=text, k=k)
    ocr_results.scores, ocr_results.idx_images, ocr_results.image_paths= ocr.text_search(text=text, k=k)

    # Concatenate lists of scores and image paths    
    list_scores = base_results.scores[0].tolist() + ocr_results.scores[0].tolist()
    list_image_paths = base_results.image_paths + ocr_results.image_paths 

    return list_scores, list_image_paths



def test_real_voting():
    text = 'Hai anh công an đang dẫn một tội phạm nam. \
            Hai anh công an mặc đồng phục ngành màu xanh lá và người tội phạm mặc áo màu hồng đỏ. \
            Ba người dàn hàng ngang và đi thẳng, tội phạm đi ở giữa.'
    os.makedirs(os.path.join(ROOT, 'results/voting'), exist_ok=True)
    list_scores, list_image_paths = sequential_process(text, 9)

    sorted_scores, sorted_paths = sort_image_paths_by_frequency_and_score(list_image_paths=list_image_paths, list_scores=list_scores)
    sorted_ids = [base.keyframes_id.index(sorted_path) for sorted_path in sorted_paths]
    df_voting = pd.DataFrame({'images_id': sorted_ids[:9], 'scores': sorted_scores[:9]})
    df_voting.to_csv(os.path.join(ROOT, 'results/voting', 'text_retrieval.csv'))
    base.show_images(sorted_paths[:9], os.path.join(ROOT, 'results/voting'), method='text')
    

def test_video_response():
    image_path_test = [
        "data/news/KeyFramesC00_V00/C00_V0001/001831.jpg",
        "data/news/KeyframeC00_V00/C00_V0000/002138.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/023979.jpg",
        "data/news/KeyFramesC00_V00/C00_V0001/016594.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/023646.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/002673.jpg",
        "data/news/KeyFramesC00_V00/C00_V0001/000676.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/027318.jpg",
        "data/news/KeyFramesC00_V00/C00_V0000/023656.jpg",
        "data/news/KeyframeC00_V00/C00_V0000/007956.jpg"
    ]
    
    video_results = video_response_processing(image_paths=image_path_test)
    print(video_results)


def timeConversion(s):
    # Write your code here
    if s[-2:] == 'PM':
        if s[:2] != '12':
            return str(int(s[:2]) + 12) + s[2:-2]
    else:
        if s[:2] == '12':
            return str(int(s[:2]) - 12) + s[2:-2]
    return s[:-2]
import json
with open(os.path.join(ROOT, 'data/dicts/info_Thuan.json'), 'r') as f:
    ocr_json = json.loads(f.read())
def ocr_ctrl_F(text):
    outputs = [value for key, value in ocr_json.items() if text in key]
    image_paths = ['data/Keyframes_' + output.split(',')[0].split('_')[0] + '/' + output.split(',')[0] +output.split(',')[1] + '.png' for output in outputs]
    return outputs, image_paths
if __name__ == '__main__':
    outputs, image_paths = ocr_ctrl_F('hello')
    # uvicorn.run('test_case:app', port=1208, host='0.0.0.0', reload=True, workers=Pool()._processes)