import clip 
import torch
from PIL import Image
import os
import glob
import tqdm
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
from pathlib import Path
import sys
import numpy as np
from PIL import ImageFile
from sentence_transformers import SentenceTransformer, util
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Read current file path
FILE = Path(__file__).resolve()
# Read folder containing file path
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
# main work directory
WORK_DIR = os.path.dirname(ROOT)
WORK_DIR = '/'.join(WORK_DIR.split('/')[:-1])

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    model, preprocess = clip.load("ViT-B/32", device=device)

    # define datasets path
    npy_folder = os.path.join('/media/hoangtv/New Volume/backup/Long', 'models/clip-v32_features_aic2023')
    path = '/media/hoangtv/New Volume/backup/data'
    if not os.path.exists(npy_folder):
        os.makedirs(npy_folder)

    video_paths = sorted(glob.glob(f"{path}/Keyframes_*/"))
    for video in tqdm.tqdm([video_paths[27]], desc='Exporting CLIP v32 visual feature'):
        # create folder to save
        npy_keyframe = os.path.join(npy_folder, video.split('/')[-2])
        if not os.path.exists(npy_keyframe):
            os.makedirs(npy_keyframe)
        print('Start from folder:', video)
        print('Save to: ', npy_keyframe)
        # if video.split('/')[-2] == 'Keyframes_L36':
        #     break
        for keyframe in sorted(os.listdir(video)):
            re_features = []
            if os.path.isdir(os.path.join(video, keyframe)):
                # save vector features
                outfile = f'{npy_keyframe}/{keyframe}.npy'
                for image_path in sorted(os.listdir(os.path.join(video, keyframe))):
                    if os.path.splitext(image_path)[1] == '.jpg':
                        image_global_path = os.path.join(video, keyframe, image_path)
                        # convert image np.array -> torch.tensor
                        image = preprocess(Image.open(image_global_path)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            # get vector features by ResNet50
                            image_feats = model.encode_image(image)
                        # normalize vector features
                        image_feats /= image_feats.norm(dim=-1, keepdim=True)
                        image_feats = image_feats.detach().cpu().numpy().astype(np.float16).flatten() 
                        re_features.append(image_feats)
 
                np.save(outfile, re_features)
if __name__ == "__main__":
    main()
    