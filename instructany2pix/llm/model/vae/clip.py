from transformers import CLIPImageProcessor,CLIPVisionModelWithProjection
from torch import nn
from PIL import Image
import os
import torch
import numpy as np
class ClipVisionModelInterfaceForLLM(nn.Module):
    def __init__(self, embed_dim,pretrained,lazy_init=True, *args, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = 1024
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(pretrained,low_cpu_mem_usage=os.environ.get('DEEPSPEED3',False) == False)

    def encode_seq(self,x,model=None):
        x = self.clip_model(torch.cat(x.pixel_values)).image_embeds
        x = x[...,None,None]
        return x,None,None

    def decode_seq(self,ind,target_shape):
        return ind
    
class ClipVisionPreprocessorForLLM:
    def __init__(self,pretrained, *args, **kwargs):
        self.processor = CLIPImageProcessor.from_pretrained(pretrained)

    def __call__(self,image):
        image = Image.open(image).convert("RGB")
        return self.processor(images=image)
    

class TensorLoader(nn.Module):
    def __init__(self, embed_dim,n_embed,lazy_init=True, *args, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed

    def encode_seq(self,x,model=None):
        # x = self.clip_model(torch.cat(x.pixel_values)).image_embeds
        # x = x[...,None,None]
        return x,None,None

    def decode_seq(self,ind,target_shape):
        return ind
    
class NpzLoader:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self,image):
        x = np.load(image)['state']
        x = torch.tensor(x)
        return x