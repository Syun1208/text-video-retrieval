import os
from PIL import Image
import numpy as np
import re
import tqdm
import time
import os
import argparse
import glob
import torch.nn as nn
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="large_coco", is_eval=True, device=device
)

vis_processors.keys()

"""
python OCR_Grand.py --test_folder /home/hoangtv/Desktop/Long/text-video-retrieval/data/news_aic2023
"""

####################### ArgumentParser ##############################
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--test_folder', default='/media/hoangtv/New Volume/backup/data', type=str, help='folder path to input images')
parser.add_argument('--root_path', default='/media/hoangtv/New Volume/backup/Nhan_CDT/AIC/model/dicts', type=str, help='folder path to input images')
args = parser.parse_args()
####################################################################

def img_caption(img_path: str):
    raw_image = Image.open(img_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    text_list = model.generate({"image": image}, use_nucleus_sampling=True, num_captions= 10, min_length=27, top_p=0.9)
    caption = ""
    for idx in range(len(text_list)):
        if idx == 0:
            caption += (text_list[idx][0].upper() + text_list[idx][1:])
        else:
            caption += (". "+text_list[idx][0].upper() + text_list[idx][1:])
    return caption




if __name__ == '__main__':
    

    root_path = args.root_path # /home/toonies/AI-Challenge/text-video-retrieval/lab/data/dicts

    root_keyframes_grand = args.test_folder # /home/toonies/AI-Challenge/text-video-retrieval/data/news_aic2023/
    # # root_keyframes = sorted(glob.glob(root_keyframes_grand+"/*"))[:3] #[.../keyframe_L01,... ,/keyframe_L10] # batch 1
    # root_keyframes = sorted(glob.glob(root_keyframes_grand+"/*"))[3:6]                                      # batch 2
    # root_keyframes = sorted(glob.glob(root_keyframes_grand+"/*"))[6:9]                                       # batch 3
    root_keyframes = sorted(glob.glob(root_keyframes_grand+"/*"))[10:]                                       # batch 4

    print("root keyframes ",root_keyframes)
    print("\nWill save in path ", root_path)
    

    with open(f"{root_path}/info_ocr.txt", 'w') as grand:
        for root_keyframe in root_keyframes:
            print("Running path: ", root_keyframe)
            path_keyframes_parent = root_keyframe.split("/")[-1] # keyframe_L01
            os.makedirs(f"{root_path}/{path_keyframes_parent}", exist_ok = True)
            with open(f"{root_path}/{path_keyframes_parent}/{path_keyframes_parent}.txt", 'w') as p:
                L0_number = sorted(glob.glob(root_keyframe+"/*"))
                for Path in tqdm.tqdm(L0_number): #.../keyframe_L01

                    start_time = time.time() 
                    
                    chars_to_replace = "^*()-_+=,\"\'?%#@!~$^&|;<>{}[]"
                    path_parent =  Path.split("/")[-2] #Keyframes_L01
                    path_child = Path.split("/")[-1] #L01_V004
                    Paths_Img = glob.glob(Path+"/*.jpg")
                    list_img_path = sorted(Paths_Img)

                    
                    with open(f"{root_path}/{path_parent}/{path_child}.txt", 'w') as f:
                        for img_path in list_img_path:
                            content_img = img_caption(img_path)
                            if content_img != "" or content_img != " ":

                                content_img = re.sub(f"[{re.escape(chars_to_replace)}]", "", content_img)
                                content_txt = [img_path.split("/")[-2],  img_path.split("/")[-1].replace(".jpg","")]
                                
                                content_txt.append(content_img)
                                list_to_str = ','.join(str(s) for s in content_txt)
                                f.write(list_to_str+"\n")
                                p.write(list_to_str+"\n")
                                grand.write(list_to_str+"\n")
                                # print("Text predict: ", list_to_str)
                            # print("Saved: ", f"{root_path}/{path_parent}/{path_child}.txt")

                        print("Saved path ", f"{root_path}/{path_parent}/{path_child}.txt")
                    print("elapsed time : {}s".format(time.time() - start_time))
    
