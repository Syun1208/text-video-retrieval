from typing import Any
import torch
from lavis.models import load_model_and_preprocess
import os
import sys
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor
from transformers import AutoModelForCausalLM
import pandas as pd
import json
import ast

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.abspath(ROOT))
WORK_DIR = os.path.dirname(ROOT)


class LAVIS:
    def __init__(self) -> None:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=self.device)

    def __call__(self, raw_image, *args: Any, **kwds: Any) -> Any:
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        # generate caption
        return self.model.generate({"image": image})
    
class HuggingFace:
    def __init__(self, checkpoint = "microsoft/git-base") -> None:
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __call__(self, image, *args: Any, **kwds: Any) -> Any:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        pixel_values = inputs.pixel_values
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_caption
    

def load_keyframes_id(path):
    with open(path, 'r') as f:
        js = json.loads(f.read())
    return js


def main():
    IP = HuggingFace()  
    data = os.path.join(WORK_DIR, 'data/news')
    txt_save = os.path.join(WORK_DIR, 'data/dicts/imgcap.txt')

    js = load_keyframes_id(os.path.join(txt_save, 'keyframes_id.json'))
    js = ast.literal_eval(js)


    info_imgcap = open(txt_save, 'w')
    for keyframe in sorted(os.listdir(data)):
        keyframe_dir = os.path.join(data, keyframe)
        for video in sorted(os.listdir(keyframe_dir)):
            video_dir = os.path.join(keyframe_dir, video)
            for image in sorted(os.listdir(video_dir)):
                image_path = os.path.join(video_dir, image)
                raw_image = Image.open(image_path).convert('RGB')
                # text
                text = IP(raw_image)
                # get id from keyframes
                image_paths = [entry['image_path'] for entry in list(js.values())]
                if image_path in image_paths:
                    image_idx = image_paths.index(image_path)
                    image_id = list(js.keys()).index(image_idx)
                
                info_imgcap.write(str(video) + '\t' + str(image_id) + '\t' + text + '\n')

    info_imgcap.close()
if __name__ == "__main__":
    main()