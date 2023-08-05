from visual_model import *
from textual_model import Textual_encoder

class PLIP_MResNet(nn.Module):
    def __init__(self, image_encoder,text_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def get_text_global_embedding(self,caption,mask):
        global_text_out = self.text_encoder.get_global_embedding(caption,mask)
        return global_text_out

    def get_image_embeddings(self,image):
        global_image_out, _,_,_,_ = self.image_encoder(image)
        return global_image_out

    def forward(self, image,text,masks):
        global_image_out,x1,x2,x3,x4 = self.image_encoder(image)
        global_text_out, part_text_out = self.text_encoder(text,masks)
        return global_image_out,x1,x2,x3,x4,global_text_out,part_text_out

def Create_PLIP_Model():
    image_encoder = Image_encoder_ModifiedResNet([3, 4, 6, 3],768 ,8 ,input_resolution=[128 ,256 ])
    text_encoder = Textual_encoder(encoder_type="bert-base-uncased")
    model = PLIP_MResNet(image_encoder, text_encoder)
    return model
    # else:
    #     raise RuntimeError(f"The image backbone you input does not meet the specification!")
