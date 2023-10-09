import requests
import json
import os
import base64
import cv2
import numpy as np

url = 'https://api.onlineocrconverter.com/api/image'
folder_path = ''
mode = 'vie'
api_key = 'ABnqoVyH_ElzjhQDynazvjW9ud2ekPxXtH-jgqw5es17_DERXoaLKe4OHqTwdgTeg54'
headers = {
    "Content-Type": "application/json",
    'key': api_key,
    'accept': 'application/json'
}

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

def export_text():
    for image_path in folder_path:
      image = cv2.imread(os.path.join(folder_path, image_path))
      im_b64 = image_to_base64(image)
      request_data = {
        'base64': im_b64,
        'language': mode
    }
    response = requests.post(url, json=request_data, headers=headers)
    if response.status_code == 200:
        # Request was successful
        response_data = response.json()
        text = response_data['text']
        # format txt

        print("Response:", response_data)
    else:
        # Request failed
        print("Request failed with status code:", response.status_code)
        print("Response:", response.text)

def test():
    image_path = '/home/hoangtv/Desktop/Long/text-video-retrieval/data/news_aic2023/Keyframes_L03/L03_V001/000563.jpg'
    image = cv2.imread(os.path.join(DATA_DIR, image_path))
    im_b64 = image_to_base64(image)
    im_cvt = base64_to_image(im_b64)

    if im_cvt is not None:
        print('Base64 is read !')

    request_data = {
        'base64': im_b64,
        'language': mode
    }
    response = requests.post(url, json=request_data, headers=headers)
    if response.status_code == 200:
        # Request was successful
        response_data = response.json()
        print("Response:", response_data)
        print('Text:', response_data['text'])
    else:
        # Request failed
        print("Request failed with status code:", response.status_code)
        print("Response:", response.text)

if __name__ == '__main__':
    test()
