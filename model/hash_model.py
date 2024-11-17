import os
import torch
import logging
import torch.nn as nn
import numpy as np
from typing import Union

from model.model import build_model
from utils import get_logger, get_summary_writer

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_uniform_(m.weight, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class LinearHash(nn.Module):

    def __init__(self, inputDim=2048, outputDim=64):
        super(LinearHash, self).__init__()
        self.fc = nn.Linear(inputDim, outputDim)
        self.fc.apply(weights_init_kaiming)
        self.drop_out = nn.Dropout(p=0.2)
    
    def forward(self, data):
        result = self.fc(data)
        return torch.tanh(self.drop_out(result))


class DCMHT(nn.Module):

    def __init__(self, 
                outputDim=64, 
                clipPath="./ViT-B-32.pt", 
                writer=None, 
                saveDir="./result/log", 
                logger: logging.Logger=None, 
                is_train=True,
                linear=False):
        super(DCMHT, self).__init__()
        os.makedirs(saveDir, exist_ok=True)
        self.logger = logger if logger is not None else get_logger(os.path.join(saveDir, "train.log" if is_train else "test.log"))
        self.writer = writer if writer is not None and is_train else get_summary_writer(os.path.join(saveDir, "tensorboard"))
        embedDim, self.clip = self.load_clip(clipPath)
        self.image_hash =  LinearHash(inputDim=embedDim, outputDim=outputDim)
        self.text_hash = LinearHash(inputDim=embedDim, outputDim=outputDim)

    def freezen(self):
        for name, param in self.clip.named_parameters():
            # print(name)
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                                        or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                # print("1")
                continue
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= 12:
                    # print("2")
                    continue
            if name.find("conv2.") == 0:
                # print("3")
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    def load_clip(self, clipPath: str) -> tuple:
        try:
            model = torch.jit.load(clipPath, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(clipPath, map_location="cpu")
        
        return state_dict["text_projection"].shape[1], build_model(state_dict)

    def encode_image(self, image):

        image_embed1 = self.clip.encode_image(image)
        image_embed2 = self.image_hash(image_embed1)

        return image_embed2
    
    def eval(self):
        self.image_hash.eval()
        self.text_hash.eval()
        # self.clip.eval()

    def train(self):
        self.image_hash.train()
        self.text_hash.train()
    
    def encode_text(self, text):

        text_embed1 = self.clip.encode_text(text)
        text_embed2 = self.text_hash(text_embed1)

        return text_embed2

    def forward(self, image, text):
        i1 = self.encode_image(image)
        t1 = self.encode_text(text)
        return i1 , t1

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import hyptorch.nn as hypnn
from hyptorch.pmath import dist_matrix


class Hier_Model(nn.Module):
    def __init__(self , enmb ):
        super(Hier_Model,self).__init__()

        self.sqence1 = nn.Sequential(
            nn.Linear(enmb , 2048),
            nn.LayerNorm(normalized_shape=(2048,),eps=1e-4,elementwise_affine=False),
            nn.Linear(2048 , 128),
            nn.Linear(128,enmb)
        )
        self.ToPoincare = hypnn.ToPoincare(c=0.1, train_x=False)
    def forward(self,enmb):
        x = self.sqence1(enmb)
        x = self.ToPoincare(x)
        x = F.normalize(x,p=2,dim=1)
        return x
#
