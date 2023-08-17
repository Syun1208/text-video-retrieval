import clip 
import torch
from PIL import Image
import os
import glob
import tqdm
import numpy as np

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # define datasets path
    npy_folder = '/home/hoangtv/Desktop/Long/text-video-retrieval/models/clip_features'
    path = '/home/hoangtv/Desktop/Long/text-video-retrieval/data/news'
    if not os.path.exists(npy_folder):
        os.makedirs(npy_folder)

    video_paths = sorted(glob.glob(f"{path}/*/"))
    for video in tqdm.tqdm(video_paths, desc='Exporting CLIP visual feature'):
        for keyframe in sorted(os.listdir(video)):
            re_features = []
            for image_path in sorted(os.listdir(os.path.join(video, keyframe))):
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

            # create folder to save
            npy_keyframe = os.path.join(npy_folder, video.split('/')[-2])
            if not os.path.exists(npy_keyframe):
                os.makedirs(npy_keyframe)
            # save vector features
            outfile = f'{npy_keyframe}/{keyframe}.npy'
            np.save(outfile, re_features)
if __name__ == "__main__":
    main()