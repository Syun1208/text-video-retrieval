import torchvision.transforms as transforms
import torch
from .utils import *
import time
import os
import tqdm
from PIL import Image
import glob
import json
import numpy as np
import shutil
import faiss
import torch.backends.cudnn as cudnn
from .test_dataloader import get_loader_test
import pickle
import argparse
from pathlib import Path
from .PLIPmodel import Create_PLIP_Model

# Read current file path
FILE = Path(__file__).resolve()
# Read folder containing file path
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
# main work directory
WORK_DIR = os.path.dirname(ROOT)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/PLIP_RN50.pth.tar')
    parser.add_argument('--image_path', type=str, default='data/CUHK-PEDES/imgs')
    parser.add_argument('--test_path', type=str,
                        default='data/CUHK-PEDES/CUHK-PEDES-test.json',
                        help='path for test annotation json file')

    parser.add_argument('--plip_model', type=str, default='MResNet_BERT')
    parser.add_argument('--img_backbone', type=str, default='ModifiedResNet',
                        help="ResNet:xxx, ModifiedResNet, ViT:xxx")
    parser.add_argument('--txt_backbone', type=str, default="bert-base-uncased")
    parser.add_argument('--img_dim', type=int, default=768, help='dimension of image embedding vectors')
    parser.add_argument('--text_dim', type=int, default=768, help='dimension of text embedding vectors')
    parser.add_argument('--layers', type=list, default=[3, 4, 6, 3], help='Just for ModifiedResNet model')
    parser.add_argument('--heads', type=int, default=8, help='Just for ModifiedResNet model')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)


    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device',type=str,default="cuda:0")
    parser.add_argument('--feature_size', type=int, default=768)
    args = parser.parse_args()
    return args

def main():
    arg = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # image transformer
    transform = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize((0.357, 0.323, 0.328),
                             (0.252, 0.242, 0.239))
    ])

    # define datasets path
    npy_folder = '/home/hoangtv/Desktop/Long/text-video-retrieval/models/plip_features'
    path = '/home/hoangtv/Desktop/Long/text-video-retrieval/data/news'
    if not os.path.exists(npy_folder):
        os.makedirs(npy_folder)
    
    # define model 
    model = Create_PLIP_Model(arg).to(device)

    video_paths = sorted(glob.glob(f"{path}/*/"))
    for video in tqdm.tqdm(video_paths, desc='Exporting PLIP visual feature'):
        video_glob = os.path.join(video, video.split('/')[-2])
        for keyframe in sorted(os.listdir(video_glob)):
            re_features = []
            for image_path in sorted(os.listdir(os.path.join(video_glob, keyframe))):
                image_global_path = os.path.join(video_glob, keyframe, image_path)
                image = Image.open(os.path.join(image_global_path)).convert('RGB')
                # convert image np.array -> torch.tensor
                image = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    # get vector features by ResNet50
                    image_feats = model.get_image_embeddings(image)
                # normalize vector features
                image_feats /= image_feats.norm(dim=-1, keepdim=True)
                image_feats = image_feats.detach().cpu().numpy().astype(np.float16).flatten() 
                re_features.append(image_feats)

            # create folder to save
            npy_keyframe = os.path.join(npy_folder, video.split('/')[-2])
            if not os.path.exists(npy_keyframe):
                os.makedirs(npy_keyframe)
            # save vector features
            outfile = f'{npy_keyframe}/{keyframe}.npy'
            np.save(outfile, re_features)
if __name__ == "__main__":
    main()
