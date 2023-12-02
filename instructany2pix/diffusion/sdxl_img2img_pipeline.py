import torch
#from diffusers import StableDiffusionXLImg2ImgPipeline,StableDiffusionXLPipeline
from ..ddim.sdxl_pipeline import StableDiffusionXLPipeline
from diffusers.utils import load_image
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    StableDiffusionXLImg2ImgPipeline
)
from typing import Any, Dict, Optional, Tuple, Union
from torch import nn
from torch import FloatTensor,Tensor
import torch


class UnCLipXL(UNet2DConditionModel):

    def __init__(self, adapt_project_dim=[1024,1024*4,2048],adapt_project_dim_2=[768,1024*4,1280],sample_size: int | None = None, in_channels: int = 4, out_channels: int = 4, center_input_sample: bool = False, flip_sin_to_cos: bool = True, freq_shift: int = 0, down_block_types: Tuple[str] = ..., mid_block_type: str | None = "UNetMidBlock2DCrossAttn", up_block_types: Tuple[str] = ..., only_cross_attention: bool | Tuple[bool] = False, block_out_channels: Tuple[int] = ..., layers_per_block: int | Tuple[int] = 2, downsample_padding: int = 1, mid_block_scale_factor: float = 1, dropout: float = 0, act_fn: str = "silu", norm_num_groups: int | None = 32, norm_eps: float = 0.00001, cross_attention_dim: int | Tuple[int] = 1280, transformer_layers_per_block: int | Tuple[int] | Tuple[Tuple] = 1, reverse_transformer_layers_per_block: Tuple[Tuple[int]] | None = None, encoder_hid_dim: int | None = None, encoder_hid_dim_type: str | None = None, attention_head_dim: int | Tuple[int] = 8, num_attention_heads: int | Tuple[int] | None = None, dual_cross_attention: bool = False, use_linear_projection: bool = False, class_embed_type: str | None = None, addition_embed_type: str | None = None, addition_time_embed_dim: int | None = None, num_class_embeds: int | None = None, upcast_attention: bool = False, resnet_time_scale_shift: str = "default", resnet_skip_time_act: bool = False, resnet_out_scale_factor: int = 1, time_embedding_type: str = "positional", time_embedding_dim: int | None = None, time_embedding_act_fn: str | None = None, timestep_post_act: str | None = None, time_cond_proj_dim: int | None = None, conv_in_kernel: int = 3, conv_out_kernel: int = 3, projection_class_embeddings_input_dim: int | None = None, attention_type: str = "default", class_embeddings_concat: bool = False, mid_block_only_cross_attention: bool | None = None, cross_attention_norm: str | None = None, addition_embed_type_num_heads=64):
        super().__init__(sample_size, in_channels, out_channels, center_input_sample, flip_sin_to_cos, freq_shift, down_block_types, mid_block_type, up_block_types, only_cross_attention, block_out_channels, layers_per_block, downsample_padding, mid_block_scale_factor, dropout, act_fn, norm_num_groups, norm_eps, cross_attention_dim, transformer_layers_per_block, reverse_transformer_layers_per_block, encoder_hid_dim, encoder_hid_dim_type, attention_head_dim, num_attention_heads, dual_cross_attention, use_linear_projection, class_embed_type, addition_embed_type, addition_time_embed_dim, num_class_embeds, upcast_attention, resnet_time_scale_shift, resnet_skip_time_act, resnet_out_scale_factor, time_embedding_type, time_embedding_dim, time_embedding_act_fn, timestep_post_act, time_cond_proj_dim, conv_in_kernel, conv_out_kernel, projection_class_embeddings_input_dim, attention_type, class_embeddings_concat, mid_block_only_cross_attention, cross_attention_norm, addition_embed_type_num_heads)
        self.adapt_project_dim = adapt_project_dim
        self.adapt_project_dim_2 = adapt_project_dim_2
        self.add_prjection()
    
    def add_prjection(self):
        adapt_project_dim = self.adapt_project_dim
        adapt_project_dim_2 = self.adapt_project_dim_2
        self.projector_unclip = nn.Sequential(
            nn.Linear(adapt_project_dim[0],adapt_project_dim[1]),
            nn.GELU(),
            nn.Linear(adapt_project_dim[1],adapt_project_dim[2]),
        )
        self.projector_unclip2 = nn.Sequential(
            nn.Linear(adapt_project_dim_2[0],adapt_project_dim_2[1]),
            nn.GELU(),
            nn.Linear(adapt_project_dim_2[1],adapt_project_dim_2[2]),
        )
    
    def forward(self, sample: FloatTensor, timestep: Tensor | float | int, encoder_hidden_states: Tensor, class_labels: Tensor | None = None, timestep_cond: Tensor | None = None, attention_mask: Tensor | None = None, cross_attention_kwargs: Dict[str, Any] | None = None, added_cond_kwargs: Dict[str, Tensor] | None = None, down_block_additional_residuals: Tuple[Tensor] | None = None, mid_block_additional_residual: Tensor | None = None, down_intrablock_additional_residuals: Tuple[Tensor] | None = None, encoder_attention_mask: Tensor | None = None, return_dict: bool = True) -> UNet2DConditionOutput | Tuple:
        added_cond_kwargs['text_embeds'] = self.projector_unclip2(added_cond_kwargs['text_embeds'].to(encoder_hidden_states))
        encoder_hidden_states = self.projector_unclip(encoder_hidden_states)
        return super().forward(sample, timestep, encoder_hidden_states, class_labels, timestep_cond, attention_mask, cross_attention_kwargs, added_cond_kwargs, down_block_additional_residuals, mid_block_additional_residual, down_intrablock_additional_residuals, encoder_attention_mask, return_dict)
    
def process_clip(batch,clip_processor=None):
    image = batch['pixel_values']
    image = clip_processor(image)
    batch['clip_input'] = image
    return batch

def build_sdxl(pretrained = "stabilityai/stable-diffusion-xl-base-1.0",
               ckpt="/localhome/data/ckpts/jacklishufan/sdxl/"):
    unet = UnCLipXL.from_pretrained(
            ckpt, subfolder="unet"
        )
    vae = AutoencoderKL.from_pretrained(
        pretrained,
        subfolder="vae" ,
    # revision=args.revision,
    )
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        pretrained,
        vae=vae,
        unet=unet,
        #revision=args.revision,
        torch_dtype=torch.float16,
    )
    return pipeline