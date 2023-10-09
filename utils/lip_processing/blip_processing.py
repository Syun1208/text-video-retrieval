from transformers import BertTokenizer
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import requests
from pathlib import Path
import os
import tqdm
import glob
import numpy as np
import sys
from langdetect import detect, DetectorFactory
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from .blip_models.med import BertConfig, BertModel
from .blip_models.blip import create_vit, init_tokenizer, load_checkpoint

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
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
DATA_DIR = '/media/hoangtv/New Volume/backup/'
DetectorFactory.seed = 0

class BLIP(nn.Module):
    def __init__(self,
                med_config = os.path.join(WORK_DIR, 'data/dicts/med_config.json'),
                image_size = 384,
                vit = 'base',
                vit_grad_ckpt = False,
                vit_ckpt_layer = 0,
                embed_dim = 256,
                ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.vision_width = vision_width
        self.text_width = text_width
        self.itm_head = nn.Linear(text_width, 2)


    def forward(self, image, caption, match_head='itm'):

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35,
                              return_tensors="pt").to(image.device)


        if match_head=='itm':
            output = self.text_encoder(text.input_ids,
                                      attention_mask = text.attention_mask,
                                      encoder_hidden_states = image_embeds,
                                      encoder_attention_mask = image_atts,
                                      return_dict = True,
                                    )
            itm_output = self.itm_head(output.last_hidden_state[:,0,:])
            return itm_output

        elif match_head=='itc':
            text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,
                                            return_dict = True, mode = 'text')
            image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)

            sim = image_feat @ text_feat.t()
            return sim
    #get features and matching
    def get_text_features(self, caption, device):
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35,
                              return_tensors="pt").to(device)
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,
                                            return_dict = True, mode = 'text')
        # print(text_output.last_hidden_state.shape)
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)
        return text_feat
    def get_image_features(self, image):
        image_embeds = self.visual_encoder(image)
        # print(image_embeds.shape)
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)
        return image_feat

    def get_fts_img(self, image):
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        return image_embeds
    def matching(self, caption, image_embeds, image_atts, device):
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35,
                                return_tensors="pt").to(device)
        output = self.text_encoder(text.input_ids,
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,
                                        return_dict = True,
                                        )
        itm_output = self.itm_head(output.last_hidden_state[:,0,:])
        print('test', output.last_hidden_state[:,0,:])
        print('shape', output.last_hidden_state[:,0,:].shape)
        return itm_output

def blip_model(pretrained='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth',**kwargs):
    model = BLIP(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert(len(msg.missing_keys)==0)
    return model

def load_image(image_path, image_size, show_image=False, device='cuda'):
    image = Image.open(image_path)
    if show_image:
        image.resize((image_size,image_size)).show()
    transform = transforms.Compose([

        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    image = transform(image).unsqueeze(0).to(device)
    return image

def main():
    print('check')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
    image_size = 384
    #load model
    model = blip_model(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device=device)
    #get features
    caption = 'DAI HOI THE THAO DONG NAM A LAN THU 31 VIET NAM 2021'
    image = load_image(image_path = os.path.join(DATA_DIR, "data/Keyframes_L01/L01_V020/004663.jpg"), 
                    image_size=384, show_image=False, device=device)
    with torch.no_grad():
        text_feature1 = model.get_text_features(caption, device=device).squeeze(0)
        image_feat = model.get_image_features(image)
    #cosine similarity
    print(image_feat @ text_feature1.t())

def image_feature_engineering():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
    image_size = 384
    #load model
    model = blip_model(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device=device)

    # define datasets path
    npy_folder = os.path.join('/media/hoangtv/New Volume/backup/Long', 'models/blip_features_aic2023')
    path = '/media/hoangtv/New Volume/backup/data'
    if not os.path.exists(npy_folder):
        os.makedirs(npy_folder)

    video_paths = sorted(glob.glob(f"{path}/Keyframes_*/"))
    for video in tqdm.tqdm(video_paths[21:], desc='Exporting BLIP visual feature'):

         # create folder to save
        npy_keyframe = os.path.join(npy_folder, video.split('/')[-2])
        if not os.path.exists(npy_keyframe):
            os.makedirs(npy_keyframe)
        print('Start from folder:', video)
        print('Save to: ', npy_keyframe)
        for keyframe in sorted(os.listdir(video)):
            re_features = []
            if os.path.isdir(os.path.join(video, keyframe)):

                # save vector features
                outfile = f'{npy_keyframe}/{keyframe}.npy'
                for image_path in sorted(os.listdir(os.path.join(video, keyframe))):
                    if os.path.splitext(image_path)[-1] == '.jpg':
                        image_global_path = os.path.join(video, keyframe, image_path)
                        # convert image np.array -> torch.tensor
                        image = load_image(image_path = image_global_path, image_size=384, show_image=False, device=device)
                        with torch.no_grad():
                            image_feats = model.get_image_features(image)
                        # normalize vector features
                        image_feats = image_feats.detach().cpu().numpy().astype(np.float16).flatten() 
                        re_features.append(image_feats)
 
                np.save(outfile, re_features)


if __name__ == '__main__':
    main()
    # feature_engineering()