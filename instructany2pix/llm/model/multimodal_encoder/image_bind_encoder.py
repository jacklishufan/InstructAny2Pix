from typing import Any
import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
# from imagebind.models import imagebind_model
try:
    from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer
except:
    pass

class LanguageBindProcesser:

    def __init__(self,modality_transform) -> None:
        self.modality_transform = modality_transform
        self.crop_size = dict(height=224,width=224)

    def __call__(self, inputs) -> Any:
        outputs = {}
        for modality,data in inputs.items():
            outputs[modality] = self.modality_transform[modality](data)
        return outputs
    
class LanguageBindVisionTower(nn.Module):
    def __init__(self, vision_tower, args,clip_type=('image',),delay_load=False):
        super().__init__()
        # 'audio'
        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.clip_type = clip_type
        self.hidden_size = 768
        if not delay_load:
            self.load_model()
        else:
            pass
            #self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        model = LanguageBind(clip_type=self.clip_type, cache_dir='src/LanguageBind/cache_dir')
        self.modality_transform = {c: transform_dict[c](model.modality_config[c]) for c in self.clip_type}
        self.image_processor = LanguageBindProcesser(self.modality_transform)
        self.vision_tower = model
        self.vision_tower.requires_grad_(False)
        #self.hidden_size = 768
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, inputs):
        embeddings = self.vision_tower({k:v for k,v in inputs.items() if k != 'info'})
        embeddings = torch.stack(
            [embeddings[info['modality']][info['idx']] for info in inputs['info']]
        )
        return embeddings

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
     
    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    # @property
    # def hidden_size(self):
    #     return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
