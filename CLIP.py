import torch
import clip
from PIL import Image
import os
import glob
from pathlib import Path
import sys
import numpy as np
import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
WORK_DIR = os.path.dirname(ROOT)
sys.path.insert(0, WORK_DIR)


def main():
    data = os.path.join(WORK_DIR, 'data/news')
    features = os.path.join(WORK_DIR, 'features')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model, preprocess = clip.load("ViT-B/16", device=device)

    if not os.path.exists(features):
        os.makedirs(features)
    for vd_path in tqdm.tqdm(sorted(os.listdir(data)), desc='Exporting CLIP features'):
        re_feats = []
        for keyframe in sorted(os.listdir(vd_path)):
            image = preprocess(Image.open(keyframe)).unsqueeze(0).to(device)

            with torch.no_grad():
                image_feats = model.encode_image(image)

        image_feats /= image_feats.norm(dim=-1, keepdim=True)
        image_feats = image_feats.detach().cpu().numpy().astype(np.float16).flatten() 

        re_feats.append(image_feats)

        name_npy = vd_path.split('/')[-1]
        outfile = f'{features}/{name_npy}.npy'
        np.save(outfile, re_feats)

if __name__ == '__main__':
    main()
