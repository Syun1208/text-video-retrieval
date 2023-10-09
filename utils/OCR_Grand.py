from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from craft_text_detector import Craft
import requests
import json
import os
from PIL import Image
from pyvi import ViTokenizer
import cv2
import numpy as np
from underthesea import text_normalize,classify
from unidecode import unidecode
import re
import tqdm
from nltk.corpus import stopwords
import torch
import time
import os
import argparse
import glob
import warnings
warnings.filterwarnings("ignore")

torch.jit.enable_onednn_fusion(True)
torch.backends.cudnn.benchmark = True
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained']=True
config['device'] = 'cuda:1'
config['predictor']['beamsearch']=False


detector = Predictor(config)

craft = Craft(output_dir=None, crop_type="box", cuda=True)

"""
python OCR_Grand.py --test_folder /home/hoangtv/Desktop/Long/text-video-retrieval/data/news_aic2023
"""

####################### ArgumentParser ##############################
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--test_folder', default='/media/hoangtv/New Volume/backup/data', type=str, help='folder path to input images')
parser.add_argument('--root_path', default='/media/hoangtv/New Volume/backup/Nhan_CDT/AIC/data_sotuyen', type=str, help='folder path to input images')
args = parser.parse_args()
####################################################################

def ocr_img(img_path: str):
    frame = cv2.imread(img_path)
    prediction_result = craft.detect_text(frame)
    boxes = prediction_result['boxes']
    if len(boxes) > 0:
        boxes = sorted(boxes, key = lambda x:x[0][1])
    output = []
    for idx, box in enumerate(boxes):
        try:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            point1, point2, point3, point4 = box
            x, y, w, h = point1[0],point1[1],point2[0] - point1[0],point4[1]-point1[1]
            crop_img = frame[y:y+h, x:x+w]
            crop_img = Image.fromarray(crop_img)
            s = detector.predict(crop_img)
            output.append(s)
        
        except:
            output.append('')

    text = ' '.join(output)
    text = text.lower()
    return text




if __name__ == '__main__':
    

    root_path = args.root_path # /home/toonies/AI-Challenge/text-video-retrieval/lab/data/dicts

    root_keyframes_grand = args.test_folder # /home/toonies/AI-Challenge/text-video-retrieval/data/news_aic2023/
    root_keyframes = sorted(glob.glob(root_keyframes_grand+"/Keyframes_L28"))                                    

    print("root keyframes ",root_keyframes)
    print("\nWill save in path ", root_path)

    for root_keyframe in root_keyframes:
        print("Running path: ", root_keyframe)
        path_keyframes_parent = root_keyframe.split("/")[-1] # keyframe_L01
        os.makedirs(f"{root_path}/{path_keyframes_parent}", exist_ok = True)
        with open(f"{root_path}/{path_keyframes_parent}/{path_keyframes_parent}.txt", 'w') as p:
            L0_number = sorted(glob.glob(root_keyframe+"/L28_V001"))
            for Path in tqdm.tqdm(L0_number): #.../keyframe_L01
                print(Path)
                start_time = time.time() 
                
                chars_to_replace = "^*()-_+=,\"\'?%#@!~$^&|;<>{}[]"
                path_parent =  Path.split("/")[-2] #Keyframes_L01
                path_child = Path.split("/")[-1] #L01_V004
                Paths_Img = glob.glob(Path+"/*.jpg")
                list_img_path = sorted(Paths_Img)[::5]

                with open(f"{root_path}/{path_parent}/{path_child}.txt", 'w') as f:
                    for img_path in list_img_path:
                        content_img = ocr_img(img_path)
                        if content_img != "" or content_img != " ":

                            content_img = re.sub(f"[{re.escape(chars_to_replace)}]", "", content_img)
                            content_txt = [img_path.split("/")[-2],  img_path.split("/")[-1].replace(".jpg","")]
                            
                            content_txt.append(content_img)
                            list_to_str = ','.join(str(s) for s in content_txt)
                            f.write(list_to_str+"\n")
                            p.write(list_to_str+"\n")
                            # print("Text predict: ", list_to_str)
                        # print("Saved: ", f"{root_path}/{path_parent}/{path_child}.txt")

                    print("Saved path ", f"{root_path}/{path_parent}/{path_child}.txt")
                print("elapsed time : {}s".format(time.time() - start_time))




