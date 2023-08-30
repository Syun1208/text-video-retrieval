from fastapi import FastAPI
import uvicorn
import numpy as np
from typing import List
from pydantic import BaseModel
from fastapi.requests import Request, HTTPConnection
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.encoders import jsonable_encoder
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2
import base64
from multiprocessing import Pool

class UserRequest(BaseModel):
    text_query: str
    top_k: int
    image_id: int

class TestRequest(BaseModel):
    text: str
    image_path: str
    image_id: int
    k: int


def image_to_base64(image):
   
    _, buffer = cv2.imencode('.jpg', image)
    img_data = base64.b64encode(buffer)
    return img_data


app = FastAPI()
app.mount("/theme", StaticFiles(directory="theme"), name="theme")
templates = Jinja2Templates(directory="templates")

origins = [
    "http://localhost",
    "http://localhost:8090"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get('/')
async def health_check():
    return {'message': 'Hello world'}

@app.post('/text_query')
async def text_query(request: UserRequest) -> Response:
    image = cv2.imread('download.jpeg')
    image_encoded = image_to_base64(image)
    return JSONResponse(content=jsonable_encoder({
        'image_base64': image_encoded,
        'text_query': request.text_query
    }))

@app.post('/test')
async def test(request: TestRequest):
    print(request)
    response = {
        'scores': np.random.randn(10).tolist(),
        'image_paths': ['a', 'b', 'c', 'd', 'e', 'f', 'r', 'e', 't', 'p'],
        'video_paths': ['a', 'b', 'c', 'd']
    }
    return JSONResponse(content=jsonable_encoder(response))

if __name__ == "__main__":
    uvicorn.run('fast_api:app', port=8090, host='0.0.0.0', reload=True, workers=Pool()._processes)