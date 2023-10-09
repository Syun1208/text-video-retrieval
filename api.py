import sys
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import socket
import base64
import os
import numpy as np
import pandas as pd
from pathlib import Path
import heapq
import asyncio
import cv2
import json
from collections import Counter
from threading import Thread
from multiprocessing import Process, Pool
from concurrent.futures import ProcessPoolExecutor
from fastapi.responses import JSONResponse, RedirectResponse, Response, FileResponse, StreamingResponse
import io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
DATA_DIR = '/media/hoangtv/New Volume/backup/'

sys.path.append(os.path.join(ROOT, 'utils'))

from utils.faiss_processing import FaissSearch
from utils.asr_processing import ASRSearch
from utils.ocr_processing import OCRSearch
from utils.image2text_processing import ImageCaptioningSearch

origins = [
    "*"
]

app = FastAPI(title='Ho Chi Minh AI Challenge 2023 - Text-Video Retrieval',description="""<h2>Made by`UTE-AI Fluc Team`</h2>""")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_lit = FaissSearch(mode='lit', show_time_compute=False)
# base_blip = FaissSearch(annotation=os.path.join(ROOT, 'data/dicts/keyframes_id_all.json'), mode='blip', show_time_compute=False)
base_blip = FaissSearch(mode='blip', show_time_compute=False)
base_clip = FaissSearch(mode='clip', show_time_compute=False)
asr = FaissSearch(mode='lit', show_time_compute=False)
ocr = FaissSearch(annotation= os.path.join(ROOT, 'data/dicts/infor_search_hp.json'),mode='bkai', show_time_compute=False)
image_captioning = FaissSearch(mode='lit', show_time_compute=False)
base_clip_v14 = FaissSearch(mode='clip-v14', show_time_compute=False)

base_lit.load_bin_file(os.path.join(ROOT, 'models/faiss_base_lit.bin'))
base_clip.load_bin_file(os.path.join(ROOT, 'models/faiss_base_clip.bin'))
base_blip.load_bin_file(os.path.join(ROOT, 'models/blip_faiss_cosine.bin'))
asr.load_bin_file(os.path.join(ROOT, 'models/faiss_ocr_lit.bin'))
base_clip_v14.load_bin_file(os.path.join(ROOT, 'models/faiss_base_clip-v14.bin'))
ocr.load_bin_file(os.path.join(ROOT, 'models/faiss_BKAI_OCR_cosine_hp.bin'))
image_captioning.load_bin_file(os.path.join(ROOT, 'models/faiss_image_captioning_lit.bin'))


# with open(os.path.join(ROOT, 'data/dicts/info_Thuan.json'), 'r') as f:
#     ocr_json = json.loads(f.read())


class UserRequest(BaseModel):
    text: str
    image_id: int
    image_path: str
    k: int
    mode_search: str
    base64_optional: bool
    submit_name: str

file_video_submit = []
frame_idx_submit = []

class MultiThread(Thread):
    def __init__(self, target=None, args=None):
        super(MultiThread, self).__init__()
        self.target = target
        self.text = args[0]
        self.k = args[1]
        self.scores, self.idx_image, self.image_paths = None, None, None

    def run(self):
        self.scores, self.idx_image, self.image_paths = self.target(self.text, self.k)

class Results:
    scores: List[np.ndarray]
    idx_images: List[int]
    image_paths: List[str]    

def multi_processing(text, k):
    list_image_paths = []
    list_scores = []
    
    ocr_results = Results()
    asr_results = Results()
    base_results = Results()
    image_captioning_results = Results()

    with ProcessPoolExecutor() as executor:
        ocr_process = executor.submit(ocr.text_search, text, k)
        asr_process = executor.submit(asr.text_search, text, k)
        base_process = executor.submit(base_lit.text_search, text, k)
        image_captioning_process = executor.submit(image_captioning.text_search, text, k)

        ocr_results.scores, ocr_results.idx_images, ocr_results.image_paths = ocr_process.result()
        asr_results.scores, asr_results.idx_images, asr_results.image_paths = asr_process.result()
        base_results.scores, base_results.idx_images, base_results.image_paths = base_process.result()
        image_captioning_results.scores, image_captioning_results.idx_images, image_captioning_results.image_paths = image_captioning_process.result()

    # Concatenate lists of scores and image paths
    list_image_paths = ocr_results.image_paths + asr_results.image_paths + base_results.image_paths + image_captioning_results.image_paths
    list_scores = ocr_results.scores[0].tolist() + asr_results.scores[0].tolist() + base_results.scores[0].tolist() + image_captioning_results.scores[0].tolist()
    
    return list_scores, list_image_paths

# def ocr_ctrl_F(text):
#     outputs = [value for key, value in ocr_json.items() if text in key]
#     image_paths = ['data/Keyframes_' + output.split(',')[0].split('_')[0] + '/' + output.split(',')[0] +output.split(',')[1] + '.png' for output in outputs]
#     return outputs, image_paths


def sequential_process(text, k):
    list_image_paths = []
    list_scores = []
    
    ocr_results = Results()
    # asr_results = Results()
    base_results = Results()
    base_blip_results = Results()
    # image_captioning_results = Results()

    # image_captioning_results.scores, image_captioning_results.idx_images, image_captioning_results.image_paths = image_captioning.text_search(text=text, k=k)
    base_results.scores, base_results.idx_images, base_results.image_paths = base_lit.text_search(text=text, k=k)
    base_blip_results.scores, base_blip_results.idx_images, base_blip_results.image_paths = base_blip.text_search(text=text, k=k)
    ocr_results.scores, ocr_results.idx_images, ocr_results.image_paths= ocr.text_search(text=text, k=k)
    # asr_results.scores, asr_results.idx_images, asr_results.image_paths = asr.text_search(text=text, k=k)

    # Concatenate lists of scores and image paths    
    list_scores = base_results.scores[0].tolist() + ocr_results.scores[0].tolist() + base_blip_results.scores[0].tolist()
    list_image_paths = base_results.image_paths + ocr_results.image_paths + base_blip_results.image_paths

    return list_scores, list_image_paths


def sort_image_paths_by_frequency_and_score(list_image_paths, list_scores):
    counter = Counter(list_image_paths)
    sorted_paths = sorted(counter.keys(), key=lambda path: (-counter[path], -list_scores[list_image_paths.index(path)]))
    sorted_scores = [list_scores[list_image_paths.index(sorted_path)] for sorted_path in sorted_paths]
    return sorted_scores, sorted_paths

def voting_algorithm(list_scores, list_image_paths, k):
    element_counts = Counter(list_image_paths)

    top_list_image_paths = list(set([element for element, count in element_counts.most_common(k)]))
    top_list_scores = [list_scores[list_image_paths.index(image_path)] for image_path in top_list_image_paths]

    
    return top_list_scores, top_list_image_paths


def image_to_base64(image):
    """
    It takes an image, encodes it as a jpg, and then encodes it as base64
    :param image: The image to be converted to base64
    :return: A base64 encoded string of the image.
    """
    _, im_arr = cv2.imencode('.jpg', image)  
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes).decode('utf-8')


    return im_b64

def base64_to_image(im_b64):
    im_bytes = base64.b64decode(im_b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img

def video_response_processing(image_paths):
    video_results = []
    for image_path in image_paths:
        id_image_path = int(os.path.splitext(image_path.split('/')[-1])[0])
        video_database = image_path.replace('data', 'video').split('/')[:-1]
        video_paths = os.listdir(video_database)
        video_result = [video for video in video_paths if id_image_path in np.arange(os.path.splitext(video.split('/')[-1])[0].split('_')[0], os.path.splitext(video.split('/')[-1])[0].split('_')[1])]
        video_results.extend(video_result)
    
    return video_results
    

@app.get('/')
async def health_check():
    return {"message": 'Hello anh Long !'}

# @app.post('/ocr_ctrl_f')
# async def ocr_ctrl_f(request: UserRequest):
#     info_outputs, image_paths = ocr_ctrl_F(request.text)
#     info_submits = [{'videoName': info_output.split(',')[0], 'frameId': info_output.split(',')[0]} for info_output in info_outputs]
#     response = {'image_paths': image_paths, 'info_submit': info_submits}
#     return JSONResponse(content=jsonable_encoder(response), status_code=200)

@app.post('/text_search')
async def text_search(request: UserRequest):
    results = {}
    images_base64 = []

    if not request.k:
        request.k = 100

    if request.mode_search == 'blip':
        scores, images_id, image_paths = base_blip.text_search(request.text, request.k)
    elif request.mode_search == 'clip-v32':
        scores, image_id, image_paths = base_clip.text_search(request.text, request.k)
    elif request.mode_search == 'lit':
        scores, images_id, image_paths = base_lit.text_search(request.text, request.k)
    elif request.mode_search == 'ocr':
        scores, images_id, image_paths = ocr.text_search(request.text, request.k)
    elif request.mode_search == 'image captioning':
        scores, images_id, image_paths = image_captioning.text_search(request.text, request.k)
    elif request.mode_search == 'asr':
        scores, images_id, image_paths = base_lit.text_search(request.text, request.k)
    elif request.mode_search =='clip-v14':
        scores, images_id, image_paths = base_clip_v14.text_search(request.text, request.k)
    else:
        scores, images_id, image_paths = base_blip.text_search(request.text, request.k)

    file_video_submit = list(map(lambda x: x.split('/')[-2], image_paths))
    frame_idx_submit = list(map(lambda x: os.path.splitext(x)[0].split('/')[-1], image_paths))
    
    if request.base64_optional:
        for image_path in image_paths:
            image = cv2.imread(os.path.join(DATA_DIR, image_path))
            image_base64 = image_to_base64(image)
            images_base64.append(image_base64)

        results = {'scores': scores[0].tolist(), 'images_base64': images_base64, 'image_paths': image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), image_paths))}
    else:

        results = {'scores': scores[0].tolist(), 'images_base64': [], 'image_paths': image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), image_paths))}

    return JSONResponse(content=jsonable_encoder(results), status_code=200)
    

@app.post('/image_search_by_id')
async def image_search_by_id(request: UserRequest):
    results = {}
    images_base64 = []
    if not request.k:
        request.k = 9

    scores, images_id, image_paths = base_lit.image_search_by_id(image_id=request.image_id, k=request.k)
    file_video_submit = list(map(lambda x: x.split('/')[-2], image_paths))
    frame_idx_submit = list(map(lambda x: os.path.splitext(x)[0].split('/')[-1], image_paths))
    if request.base64_optional:
        for image_path in image_paths:
            image = cv2.imread(os.path.join(DATA_DIR, image_path))
            image_base64 = image_to_base64(image)
            images_base64.append(image_base64)

        results = {'scores': scores[0].tolist(), 'images_base64': images_base64, 'image_paths': image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), image_paths))}
    else:

        results = {'scores': scores[0].tolist(), 'images_base64': [], 'image_paths': image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), image_paths))}    

    return JSONResponse(content=jsonable_encoder(results))

@app.post('/image_search_by_path')
async def image_search_by_path(request: UserRequest):
    results = {}
    images_base64 = []
    if not request.k:
        request.k = 9

    scores, images_id, image_paths = base_lit.image_search_by_path(image_id=request.image_path, k=request.k)
    file_video_submit = list(map(lambda x: x.split('/')[-2], image_paths))
    frame_idx_submit = list(map(lambda x: os.path.splitext(x)[0].split('/')[-1], image_paths))
    if request.base64_optional:
        for image_path in image_paths:
            image = cv2.imread(os.path.join(DATA_DIR, image_path))
            image_base64 = image_to_base64(image)
            images_base64.append(image_base64)

        results = {'scores': scores[0].tolist(), 'images_base64': images_base64, 'image_paths': image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), image_paths))}
    else:

        results = {'scores': scores[0].tolist(), 'images_base64': [], 'image_paths': image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), image_paths))}    
    return JSONResponse(content=jsonable_encoder(results))

@app.post('/asr_search')
async def asr_search(request: UserRequest):
    results = {}
    images_base64 = []
    if not request.k:
        request.k = 9

    scores, idx_image, image_paths = asr.text_search(text=request.text, k=request.k)
    file_video_submit = list(map(lambda x: x.split('/')[-2], image_paths))
    frame_idx_submit = list(map(lambda x: os.path.splitext(x)[0].split('/')[-1], image_paths))
    if request.base64_optional:
        for image_path in image_paths:
            image = cv2.imread(os.path.join(DATA_DIR, image_path))
            image_base64 = image_to_base64(image)
            images_base64.append(image_base64)

        results = {'scores': scores[0].tolist(), 'images_base64': images_base64, 'image_paths': image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), image_paths))}
    else:

        results = {'scores': scores[0].tolist(), 'images_base64': [], 'image_paths': image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), image_paths))}
    return JSONResponse(content=jsonable_encoder(results))


@app.post('/ocr_search')
async def ocr_search(request: UserRequest):
    results = {}
    images_base64 = []

    if not request.k:
        request.k = 9

    scores, idx_image, image_paths = ocr.text_search(text=request.text, k=request.k)
    file_video_submit = list(map(lambda x: x.split('/')[-2], image_paths))
    frame_idx_submit = list(map(lambda x: os.path.splitext(x)[0].split('/')[-1], image_paths))
    if request.base64_optional:
        for image_path in image_paths:
            image = cv2.imread(os.path.join(DATA_DIR, image_path))
            image_base64 = image_to_base64(image)
            images_base64.append(image_base64)

        results = {'scores': scores[0].tolist(), 'images_base64': images_base64, 'image_paths': image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), image_paths))}
    else:

        results = {'scores': scores[0].tolist(), 'images_base64': [], 'image_paths': image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), image_paths))}
    return JSONResponse(content=jsonable_encoder(results))

@app.post('/image2text_search')
async def image2text_search(request: UserRequest):
    results = {}
    images_base64 = []

    if not request.k:
        request.k = 9

    scores, idx_image, image_paths = image_captioning.text_search(text=request.text, k=request.k)
    file_video_submit = list(map(lambda x: x.split('/')[-2], image_paths))
    frame_idx_submit = list(map(lambda x: os.path.splitext(x)[0].split('/')[-1], image_paths))
    if request.base64_optional:
        for image_path in image_paths:
            image = cv2.imread(os.path.join(DATA_DIR, image_path))
            image_base64 = image_to_base64(image)
            images_base64.append(image_base64)

        results = {'scores': scores[0].tolist(), 'images_base64': images_base64, 'image_paths': image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), image_paths))}
    else:

        results = {'scores': scores[0].tolist(), 'images_base64': [], 'image_paths': image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), image_paths))}
    return JSONResponse(content=jsonable_encoder(results))

@app.post('/image2text_ocr_search')
async def image2text_ocr_search(request: UserRequest):
    results = {}
    images_base64 = []
    if not request.k:
        request.k = 9

    scores_ic, idx_image_ic, image_paths_ic = image_captioning.text_search(text=request.text, k=request.k)
    scores_ocr, idx_image_ocr, image_paths_ocr = ocr.text_search(text=request.text, k=request.k)

    list_scores = scores_ic[0].tolist() + scores_ocr[0].tolist()
    list_image_paths = image_paths_ic + image_paths_ocr

    max_scores, max_image_paths = sort_image_paths_by_frequency_and_score(list_image_paths, list_scores)
    file_video_submit = list(map(lambda x: x.split('/')[-2], max_image_paths))
    frame_idx_submit = list(map(lambda x: os.path.splitext(x)[0].split('/')[-1], max_image_paths))
    if request.base64_optional:
        for image_path in max_image_paths:
            image = cv2.imread(os.path.join(DATA_DIR, image_path))
            image_base64 = image_to_base64(image)
            images_base64.append(image_base64)

        results = {'scores': max_scores[0].tolist(), 'images_base64': images_base64, 'image_paths': max_image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), max_image_paths))}
    else:
        results = {'scores': max_scores[0].tolist(), 'images_base64': [], 'image_paths': max_image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), max_image_paths))}    
    return JSONResponse(content=jsonable_encoder(results))

@app.post('/asr_ocr_search')
async def asr_ocr_search(request: UserRequest):
    results = {}
    images_base64 = []
    if not request.k:
        request.k = 9

    scores_asr, idx_image_asr, image_paths_asr = asr.text_search(text=request.text, k=request.k)
    scores_ocr, idx_image_ocr, image_paths_ocr = ocr.text_search(text=request.text, k=request.k)

    list_scores = scores_asr[0].tolist() + scores_ocr[0].tolist()
    list_image_paths = image_paths_asr + image_paths_ocr

    max_scores, max_image_paths = sort_image_paths_by_frequency_and_score(list_image_paths, list_scores)
    file_video_submit = list(map(lambda x: x.split('/')[-2], max_image_paths))
    frame_idx_submit = list(map(lambda x: os.path.splitext(x)[0].split('/')[-1], max_image_paths))
    if request.base64_optional:
        for image_path in max_image_paths:
            image = cv2.imread(os.path.join(DATA_DIR, image_path))
            image_base64 = image_to_base64(image)
            images_base64.append(image_base64)

        results = {'scores': max_scores[0].tolist(), 'images_base64': images_base64, 'image_paths': max_image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), max_image_paths))}
    else:

        results = {'scores': max_scores[0].tolist(), 'images_base64': [], 'image_paths': max_image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), max_image_paths))}    
    return JSONResponse(content=jsonable_encoder(results))

@app.post('/image2text_asr_search')
async def image2text_asr_search(request: UserRequest):
    results = {}
    images_base64 = []
    if not request.k:
        request.k = 9

    scores_asr, idx_image_asr, image_paths_asr = asr.text_search(text=request.text, k=request.k)
    scores_ic, idx_image_ic, image_paths_ic = image_captioning.text_search(text=request.text, k=request.k)

    list_scores = scores_asr[0].tolist() + scores_ic[0].tolist()
    list_image_paths = image_paths_asr + image_paths_ic

    max_scores, max_image_paths = sort_image_paths_by_frequency_and_score(list_image_paths, list_scores)
    file_video_submit = list(map(lambda x: x.split('/')[-2], max_image_paths))
    frame_idx_submit = list(map(lambda x: os.path.splitext(x)[0].split('/')[-1], max_image_paths))
    if request.base64_optional:
        for image_path in max_image_paths:
            image = cv2.imread(os.path.join(DATA_DIR, image_path))
            image_base64 = image_to_base64(image)
            images_base64.append(image_base64)

        results = {'scores': max_scores[0].tolist(), 'images_base64': images_base64, 'image_paths': max_image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), max_image_paths))}
    else:

        results = {'scores': max_scores[0].tolist(), 'images_base64': [], 'image_paths': max_image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), max_image_paths))}    
    return JSONResponse(content=jsonable_encoder(results))


@app.post('/combine_search')
async def combine_search(request: UserRequest):
    results = {}
    images_base64 = []

    if not request.k:
        request.k = 9
    list_scores, list_image_paths = sequential_process(text=request.text, k=request.k)
    # Remove duplicated image paths
    list_image_paths_nonduplicate = list(set(list_image_paths))
    list_idx_image_paths = [list_image_paths.index(string) for string in list_image_paths_nonduplicate]
    list_image_paths = list_image_paths_nonduplicate
    list_scores = [list_scores[idx] for idx in list_idx_image_paths]
    # Sort concatenated lists
    list_scores = sorted(list_scores, reverse=True)
    idx_sorted_scores = [i for i, x in sorted(enumerate(list_scores), key=lambda x: x[1], reverse=True)]
    list_image_paths = [list_image_paths[i] for i in idx_sorted_scores]
    # Find top K maximum of the sorted lists
    max_scores, max_image_paths = list_scores[:request.k], list_image_paths[:request.k]
    file_video_submit = list(map(lambda x: x.split('/')[-2], max_image_paths))
    frame_idx_submit = list(map(lambda x: os.path.splitext(x)[0].split('/')[-1], max_image_paths))
    if request.base64_optional:
        for image_path in max_image_paths:
            image = cv2.imread(os.path.join(DATA_DIR, image_path))
            image_base64 = image_to_base64(image)
            images_base64.append(image_base64)

        results = {'scores': max_scores[0].tolist(), 'images_base64': images_base64, 'image_paths': max_image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), max_image_paths))}
    else:

        results = {'scores': max_scores[0].tolist(), 'images_base64': [], 'image_paths': max_image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), max_image_paths))}
    return JSONResponse(content=jsonable_encoder(results))


@app.post('/vote_search')
async def vote_search(request: UserRequest):
    results = {}
    images_base64 = []

    if not request.k:
        request.k = 9

    list_scores, list_image_paths = sequential_process(text=request.text, k=request.k)

    max_scores, max_image_paths = voting_algorithm(list_scores, list_image_paths, request.k)
    file_video_submit = list(map(lambda x: x.split('/')[-2], max_image_paths))
    frame_idx_submit = list(map(lambda x: os.path.splitext(x)[0].split('/')[-1], max_image_paths))
    if request.base64_optional:
        for image_path in max_image_paths:
            image = cv2.imread(os.path.join(DATA_DIR, image_path))
            image_base64 = image_to_base64(image)
            images_base64.append(image_base64)

        results = {'scores': max_scores, 'images_base64': images_base64, 'image_paths': max_image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), max_image_paths))}
    else:

        results = {'scores': max_scores, 'images_base64': [], 'image_paths': max_image_paths, 'video_path': list(map(lambda x: os.path.join('video/news_aic2023', x.split('/')[-3], x.split('/')[-2] + '.mp4'), max_image_paths))}
    return JSONResponse(content=jsonable_encoder(results))

@app.post('/place_recognition_search')
async def place_recognition_search(request: UserRequest):
    results = {}
    images_base64 = []
    if not request.k:
        request.k = 9


@app.post('/submit', response_class=FileResponse)
async def submit(request: UserRequest):
    file_response = os.path.join(ROOT, f'{request.submit_name}.csv')
    dict_results = {'file_videos': file_video_submit, 'frames_idx': frame_idx_submit}
    results = pd.DataFrames(dict_results)
    results.to_csv(file_response, index = False, header=False)
    response = FileResponse(file_response, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=downloaded_file.csv"
    return response


def test():

    text = 'Khung cảnh trong một sự kiện sản xuất khí cầu tại venezuela. \
        Có 3 khí cầu to lớn đang chuẩn bị được bay lên bầu trời. \
        Chuyển cảnh cuộc phổng vấn một nhóm người đang chuẩn bị trải nghiệm với một khí cầu. \
        Đại diện cho nhóm là người đàn ông đeo kính khoác một chiếc áo đỏ đen trả lời phỏng vấn về sự kiện này.'
    k = 9

    # start_multi = time.time()
    # list_scores, list_image_paths = multi_processing(text, k)
    # # Remove duplicated image paths
    # list_image_paths_nonduplicate = list(set(list_image_paths))
    # list_idx_image_paths = [list_image_paths.index(string) for string in list_image_paths_nonduplicate]
    # list_scores = [list_scores[idx] for idx in list_idx_image_paths]
    # # Sort concatenated lists
    # list_scores = sorted(list_scores, reverse=True)
    # idx_sorted_scores = [i for i, x in sorted(enumerate(list_scores), key=lambda x: x[1], reverse=True)]
    # list_image_paths = [list_image_paths[i] for i in idx_sorted_scores]
    # # Find top K maximum of the sorted lists
    # max_scores, max_image_paths = list_scores[:k], list_image_paths[:k]
    # results = {'scores': max_scores, 'image_paths': max_image_paths}
    # print('Time of multi-processing: ', time.time() - start_multi)

    start_sq = time.time()
    list_scores, list_image_paths = sequential_process(text, k)
    # Remove duplicated image paths
    list_image_paths_nonduplicate = list(set(list_image_paths))
    list_idx_image_paths = [list_image_paths.index(string) for string in list_image_paths_nonduplicate]
    list_image_paths = list_image_paths_nonduplicate
    list_scores = [list_scores[idx] for idx in list_idx_image_paths]
    # Sort concatenated lists
    list_scores = sorted(list_scores)
    idx_sorted_scores = [i for i, x in sorted(enumerate(list_scores), key=lambda x: x[1])]
    list_image_paths = [list_image_paths[i] for i in idx_sorted_scores]
    # Find top K maximum of the sorted lists
    max_scores, max_image_paths = list_scores[:k], list_image_paths[:k]
    results = {'scores': max_scores, 'image_paths': max_image_paths}
    print('Time of sequential process: ', time.time() - start_sq)
    
if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0', port=8090, reload=True, workers=Pool()._processes)

