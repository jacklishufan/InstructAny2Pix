import pathlib
import os
from typing import Any
from .image_vqvae import instantiate_from_config,ImageVAEProcesser
from .audio_vqvae import AudioVAEProcesser
from .clip import ClipVisionPreprocessorForLLM,NpzLoader
import yaml
import torch
from torch import nn
CURR_PATH = pathlib.Path(__file__).parent.resolve()

def build_vae(config,modality):
    config_file = CURR_PATH / f'{config}.yaml'
    with open(config_file) as f:
        config = yaml.safe_load(f.read())
    model_config  =  config['model']
    model = instantiate_from_config(model_config)
    ckpt = config['ckpt']
    if ckpt:
        sd = torch.load(ckpt,map_location='cpu')["state_dict"]
        model.load_state_dict(sd, strict=False)
    if modality == 'image':
        if config.get("processor") == 'clip':
            processor = ClipVisionPreprocessorForLLM(pretrained=config['model']['params']['pretrained'])
        else:
            processor = ImageVAEProcesser(config['image_size'])
    elif modality == 'audio':
        if config.get("processor") == 'npz':
            processor = NpzLoader()
        else:
            processor = AudioVAEProcesser(config['data']['params']['sample_rate'])
    return model,processor



class VQVAEProcessor:
    def __init__(self,image_vae_processor,audio_vae_processor) -> None:
        self.image_vae_processor = image_vae_processor
        self.audio_vae_processor = audio_vae_processor

    def __call__(self, x,modality) -> Any:
        if modality == 'image':
            return self.image_vae_processor(x)
        elif modality == 'audio':
            return self.audio_vae_processor(x)
        else:
            raise NotImplemented

class VQVAE(nn.Module):

    def __init__(self,image_vae,audio_vae) -> None:
        super().__init__()
        self.image_vae_processor = None
        self.audio_vae_processor = None
        self.embed_dim_image = 1
        self.embed_dim_audio = 1
        self.vocab_size_image = 1
        self.vocab_size_audio = 1
        if image_vae:
            self.image_vae,self.image_vae_processor = build_vae(image_vae,'image')
            self.image_vae.requires_grad_(False)
            self.embed_dim_image = self.image_vae.embed_dim
            self.vocab_size_image = self.image_vae.n_embed
        if audio_vae:
            self.audio_vae,self.audio_vae_processor = build_vae(audio_vae,'audio')
            self.audio_vae.requires_grad_(False)
            self.embed_dim_audio = self.audio_vae.embed_dim
            self.vocab_size_audio = self.audio_vae.n_embed
        self.processor = VQVAEProcessor(self.image_vae_processor,self.audio_vae_processor)

    @torch.no_grad()
    def forward(self,x):
        out = {}
        if 'image' in x:
            out['image'] = self.image_vae.encode_seq(x['image'])
        if 'audio' in x:
            out['audio'] = self.audio_vae.encode_seq(x['audio'])
        return out
    
