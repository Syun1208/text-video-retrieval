from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import socket
import base64
from fastapi.responses import JSONResponse, RedirectResponse, Response
from utils.faiss_processing import FaissSearch
from utils.asr_processing import ASRSearch
from utils.ocr_processing import OCRSearch
from utils.image2text_processing import ImageCaptioningSearch
from threading import Thread
from multiprocessing import Process

import os
import numpy as np
from pathlib import Path
from fastapi.encoders import jsonable_encoder
import sys
import cv2

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

app = FastAPI()

base = FaissSearch(mode='lit')
asr = ASRSearch(mode='lit')
ocr = OCRSearch(mode='lit')
image_captioning = ImageCaptioningSearch(mode='lit')

base.load_bin_file()
asr.load_bin_file()
ocr.load_bin_file()
image_captioning.load_bin_file()

class UserRequest(BaseModel):
    text: str
    image_id: int
    k: int
    mode_search: str


class MultiProcess(Thread):
    def __init__(self, target=None, args=None):
        super(MultiProcess, self).__init__()
        self.target = target
        self.text = args[0]
        self.k = args[1]
        self.scores, self.idx_image, self.image_paths = None, None, None

    def run(self):
        self.scores, self.idx_image, self.image_paths = self.target(self.text, self.k)

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


@app.get('/')
async def health_check():
    return {"message": 'Hello anh Long !'}

@app.post('/text_search')
async def text_search(request: UserRequest):
    results = {}

    scores, images_id, image_paths = base.text_search(request.text, request.k)
    
    results = {'scores': scores[0].tolist(), 'image_path': image_paths}

    return JSONResponse(content=jsonable_encoder(results))
    

@app.post('/image_search')
async def image_search(request: UserRequest):
    results = {}

    scores, images_id, image_paths = base.image_search(image_id=request.image_id, k=request.k)

    results = {'scores': scores[0].tolist(), 'image_paths': image_paths}
    
    return JSONResponse(content=jsonable_encoder(results))

@app.post('/video_search')
async def video_search(request: UserRequest):
    pass

@app.post('/asr_search')
async def asr_search(request: UserRequest):
    results = {}

    scores, idx_image, image_paths = asr.text_search(text=request.text, k=request.k)

    results = {'scores': scores[0].tolist(), 'image_paths': image_paths}

    return JSONResponse(content=jsonable_encoder(results))


@app.post('/ocr_search')
async def ocr_search(request: UserRequest):
    results = {}

    scores, idx_image, image_paths = ocr.text_search(text=request.text, k=request.k)

    results = {'scores': scores[0].tolist(), 'image_paths': image_paths}

    return JSONResponse(content=jsonable_encoder(results))

@app.post('/image2text_search')
async def image2text_search(request: UserRequest):
    results = {}

    scores, idx_image, image_paths = image_captioning.text_search(text=request.text, k=request.k)

    results = {'scores': scores[0].tolist(), 'image_paths': image_paths}

    return JSONResponse(content=jsonable_encoder(results))


@app.post('/combine_search')
async def combine_search(request: UserRequest):
    list_image_paths = []
    list_scores = []

    ocr_process = MultiProcess(target=ocr.text_search, args=(request.text, request.k))
    asr_process = MultiProcess(target=asr.text_search, args=(request.text, request.k))
    base_process = MultiProcess(target=base.text_search, args=(request.text, request.k))
    image_captioning_process = MultiProcess(target=image_captioning.text_search, args=(request.text, request.k))

    ocr_process.start()
    asr_process.start()
    base_process.start()
    image_captioning_process.start()

    ocr_process.join()
    asr_process.join()
    base_process.join()
    image_captioning_process.join()

    # Concatenate lists of scores and image paths
    list_image_paths = ocr_process.image_paths + asr_process.image_paths + base_process.image_paths + image_captioning_process.image_paths
    list_scores = ocr_process.scores[0].tolist() + asr_process.scores[0].tolist() + base_process.scores[0].tolist() + image_captioning_process.scores[0].tolist()
    # Remove duplicated image paths
    list_image_paths_nonduplicate = list(set(list_image_paths))
    list_idx_image_paths = [list_image_paths.index(string) for string in list_image_paths_nonduplicate]
    list_scores = [list_scores[idx] for idx in list_idx_image_paths]
    # Sort concatenated lists
    list_scores = sorted(list_scores)
    idx_sorted_scores = [i for i, x in sorted(enumerate(list_scores), key=lambda x: x[1])]
    list_image_paths = [list_image_paths[i] for i in idx_sorted_scores]
    # Find top K maximum of the sorted lists
    max_scores, max_image_paths = list_scores[:request.k], list_image_paths[:request.k]
    results = {'scores': max_scores, 'image_paths': max_image_paths}

    return JSONResponse(content=jsonable_encoder(results))

@app.post('/vote_search')
async def vote_search(request: UserRequest):
    pass


@app.get('/submit')
async def submit():
    pass


def test():

    text = 'Khung cảnh trong một sự kiện sản xuất khí cầu tại venezuela. \
        Có 3 khí cầu to lớn đang chuẩn bị được bay lên bầu trời. \
        Chuyển cảnh cuộc phổng vấn một nhóm người đang chuẩn bị trải nghiệm với một khí cầu. \
        Đại diện cho nhóm là người đàn ông đeo kính khoác một chiếc áo đỏ đen trả lời phỏng vấn về sự kiện này.'
    k = 9
    list_image_paths = []
    list_scores = []

    ocr_process = MultiProcess(target=ocr.text_search, args=(text, k))
    asr_process = MultiProcess(target=asr.text_search, args=(text, k))
    base_process = MultiProcess(target=base.text_search, args=(text, k))
    image_captioning_process = MultiProcess(target=image_captioning.text_search, args=(text, k))

    ocr_process.start()
    asr_process.start()
    base_process.start()
    image_captioning_process.start()

    ocr_process.join()
    asr_process.join()
    base_process.join()
    image_captioning_process.join()

    # Concatenate lists of scores and image paths
    print(ocr_process.image_paths , asr_process.image_paths , base_process.image_paths , image_captioning_process.image_paths)
    list_image_paths = ocr_process.image_paths + asr_process.image_paths + base_process.image_paths + image_captioning_process.image_paths
    list_scores = ocr_process.scores[0].tolist() + asr_process.scores[0].tolist() + base_process.scores[0].tolist() + image_captioning_process.scores[0].tolist()
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
    results = {'scores': max_scores, 'image_paths': max_image_paths}
    print(results)


if __name__ == '__main__':
    # uvicorn.run('api:app', host='0.0.0.0', port=8090, reload=True, workers=2)
    test()
