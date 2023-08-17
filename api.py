from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import socket
import base64
from utils.faiss_processing import FaissSearch
from utils.text2image_processing import ImageCaptioning
import os
from pathlib import Path
from fastapi.responses import ORJSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import sys
import cv2

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
app.mount("/static", StaticFiles(directory="templates"), name="static")
templates = Jinja2Templates(directory="templates")


class Request:
    text: str
    image_id: str
    k: int
    mode_search: str

def image_to_base64(image):
    """
    It takes an image, encodes it as a jpg, and then encodes it as base64
    :param image: The image to be converted to base64
    :return: A base64 encoded string of the image.
    """
    _, buffer = cv2.imencode('.jpg', image)
    img_data = base64.b64encode(buffer)
    return img_data

@app.post('/')
async def health_check():
    return {"message": 'Hello anh Nong !'}

@app.post('/text_search', response_class=HTMLResponse)
async def text_search(request: Request):
    # text from user
    text = request.text
    list_image_encoded = []
    # Create an object vector search
    faiss_search = FaissSearch(mode=request.mode_search)
    faiss_search.load_bin_file()
    scores, images_id, image_paths = faiss_search.text_search(text, k=9)
    # Encoder images to base64
    for image_path in image_paths:
        image = cv2.imread(WORK_DIR, image_path)
        image_encoded = image_to_base64(image)
        list_image_encoded.append(image_encoded)
    
    results = {'scores': scores, 'image_encoded': list_image_encoded}
    return templates.TemplateResponse("index.html", {"request": request, **results})
    

@app.post('/image_search', response_class=HTMLResponse)
async def image_search(request: Request):
    # text from user
    image_id = request.image_id
    list_image_encoded = []
    # Create an object vector search
    faiss_search = FaissSearch(mode=request.mode_search)
    faiss_search.load_bin_file()
    scores, images_id, image_paths = faiss_search.image_search(image_id=image_id, k=9)
    # Encoder images to base64
    for image_path in image_paths:
        image = cv2.imread(WORK_DIR, image_path)
        image_encoded = image_to_base64(image)
        list_image_encoded.append(image_encoded)
    
    results = {'scores': scores, 'image_encoded': list_image_encoded}
    return templates.TemplateResponse("index.html", {"request": request, **results})

@app.post('/video_search')
async def video_search():
    pass

@app.post('/asr_search')
async def asr_search():
    pass

@app.post('/ocr_search')
async def ocr_search():
    pass

@app.post('/image2text_search')
async def image2text_search():
    pass

@app.post('/combine_search')
async def combine_search():
    pass

@app.get('/submit')
async def submit():
    pass

if __name__ == '__main__':
    uvicorn.run('api:app', host=HOST, port=8008, reload=True)