from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
import time
import uvicorn
import numpy as np
import socket
import base64
import sorting
from threading import Thread
from multiprocessing import Process
import threading
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.encoders import jsonable_encoder
# from utils.asr_processing import ASRSearch

app = FastAPI()

class UserRequest(BaseModel):
    image_paths: List[str]
    scores: List[float]

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs, Verbose)
        self._return = None
    def run(self):
        if self._Thread__target is not None:
            self._return = self._Thread__target(*self._Thread__args,
                                                **self._Thread__kwargs)
    def join(self):
        Thread.join(self)
        return self._return
    

class ProcessTestReturnValue(Thread):
    def __init__(self, target=None, args=None):
        super(ProcessTestReturnValue, self).__init__()
        self.target = target
        self.scores, self.image_paths, self.text, self.k = None, None, None, None
        self.args = args
    def run(self):
        self.scores, self.image_paths, self.text, self.k = self.target(self.args[0], self.args[1])
    

@app.post("/test")
async def test(request: UserRequest):
    results = {'scores': [0.53, 0.53, 0.46, 0.89, 0.98], 'image_paths': ['afasf', 'afasfa', 'afdaf', 'asfasf', 'afaf']}
    return JSONResponse(content=jsonable_encoder(results))

def test1(text, k):
    scores = [0.53, 0.53, 0.46, 0.89, 0.98]
    image_paths = ['afasf', 'afasfa', 'afdaf', 'asfasf', 'afaf']
    return scores, image_paths, text, k
def test2(text, k):
    scores = [0.53, 0.53, 0.46, 0.89, 0.98]
    image_paths = ['afasf', 'afasfa', 'afdaf', 'asfasf', 'afaf']
    return scores, image_paths, text, k

def test_findind_topk_x3():
    k = 10
    list_scores = [
    0.2077348679304123,
    0.20756712555885315,
    0.20677229762077332,
    0.20454947650432587,
    0.20420105755329132,
    0.20369459688663483,
    0.20265847444534302,
    0.2019471526145935,
    0.2012714445590973,
    0.2012714445590973,
    0.2012714445590973
    ]
    list_image_paths = [
    "data/news/KeyFramesC00_V00/C00_V0000/025543.jpg",
    "data/news/KeyFramesC00_V00/C00_V0000/020020.jpg",
    "data/news/KeyFramesC00_V00/C00_V0000/025718.jpg",
    "data/news/KeyFramesC00_V00/C00_V0000/025780.jpg",
    "data/news/KeyFramesC00_V00/C00_V0000/019971.jpg",
    "data/news/KeyFramesC00_V00/C00_V0000/018828.jpg",
    "data/news/KeyFramesC00_V00/C00_V0000/008928.jpg",
    "data/news/KeyFramesC00_V00/C00_V0000/025622.jpg",
    "data/news/KeyFramesC00_V00/C00_V0000/018948.jpg",
    "data/news/KeyFramesC00_V00/C00_V0000/025543.jpg",
    "data/news/KeyFramesC00_V00/C00_V0000/025543.jpg"
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
    print('Time: ', time.time() - start)
    # list test cases
    scores_test =  [
    0.2077348679304123,
    0.20756712555885315,
    0.20677229762077332,
    0.20454947650432587,
    0.20420105755329132,
    0.20369459688663483,
    0.20265847444534302,
    0.2019471526145935,
    0.2012714445590973
    ]
    image_path_test = [
    "data/news/KeyFramesC00_V00/C00_V0000/025543.jpg",
    "data/news/KeyFramesC00_V00/C00_V0000/020020.jpg",
    "data/news/KeyFramesC00_V00/C00_V0000/025718.jpg",
    "data/news/KeyFramesC00_V00/C00_V0000/025780.jpg",
    "data/news/KeyFramesC00_V00/C00_V0000/019971.jpg",
    "data/news/KeyFramesC00_V00/C00_V0000/018828.jpg",
    "data/news/KeyFramesC00_V00/C00_V0000/008928.jpg",
    "data/news/KeyFramesC00_V00/C00_V0000/025622.jpg",
    "data/news/KeyFramesC00_V00/C00_V0000/018948.jpg"
    ]
    print(max_scores, max_image_paths)
    print(max_scores == sorted(scores_test),  max_image_paths == image_path_test)

def test_pop():
    start = time.time()
    scores, image_paths, text, k = test1('hello world', 3)
    print(scores, image_paths, text, k)
    scores, image_paths, text, k = test1('hi', 4)
    print(scores, image_paths, text, k)
    print('Time POP: ', time.time() - start)
def test_multi_process():
    start = time.time()
    process_1 = ProcessTestReturnValue(target=test1, args=('hello world', 3))
    process_1.start()
    process_2 = ProcessTestReturnValue(target=test2, args=('hi', 4))
    process_2.start()

    process_1.join()
    process_2.join()

    scores, image_paths, text, k = process_1.scores, process_1.image_paths, process_1.text, process_1.k
    print(scores, image_paths, text, k)
    scores, image_paths, text, k = process_2.scores, process_2.image_paths, process_2.text, process_2.k
    print(scores, image_paths, text, k)
    print('Time: ', time.time() - start)
if __name__ == '__main__':
    test_pop()
    test_multi_process()