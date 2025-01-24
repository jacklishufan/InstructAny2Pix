# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers

from instructany2pix.llm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,DEFAULT_IM_GEN_TOKEN,DEFAULT_AUDIO_GEN_TOKEN,DEFAULT_MSK_TOKEN,DEFAULT_IM_GEN_START_TOKEN,DEFAULT_AUDIO_GEN_START_TOKEN,DEFAULT_AUDIO_TOKEN
from torch.utils.data import Dataset
from instructany2pix.training.trainer import LLaVATrainer

from instructany2pix.llm import conversation as conversation_lib
from instructany2pix.llm.model import *
from instructany2pix.llm.mm_utils import tokenizer_image_token

from PIL import Image
import torch.distributed as dist

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    mm_use_gen: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="projection")
    vae_image: Optional[str] = field(default=None)
    vae_audio: Optional[str] = field(default=None)
    dev: Optional[str] = field(default=None)
    load: Optional[str] = None
    


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    output_text: Optional[bool] = False
    media_map: str = 'local/npz_files.json'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    group_by_modality_length: bool = field(default=False)
    split_loading: bool = False


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str,
                                   force_save=False):
    """Collects the state dict and dump to disk."""
    
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save or force_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg



def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources





def preprocess_plain_gen(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image:bool,
    info:Dict,
) -> Dict:
    # add end signal and concatenate together
    #conv = conversation_lib.default_conversation.copy()
    conv = conversation_lib.conv_templates["vicuna_v1"].copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            val = sentence['value']
            val = val.replace(DEFAULT_IM_GEN_TOKEN,DEFAULT_IM_GEN_START_TOKEN + DEFAULT_IM_GEN_TOKEN*info['generation_seq_len'])
            val = val.replace(DEFAULT_AUDIO_GEN_TOKEN,DEFAULT_AUDIO_GEN_START_TOKEN+DEFAULT_AUDIO_GEN_TOKEN*info['generation_seq_len'])

            conv.append_message(role, val)
        assert conv.sep is not None and conv.sep2 is not None
        conversations.append(conv.get_prompt())
    
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                breakpoint()
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. {rounds}"
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    info={},
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # hack
    if info.get('generation'):
        return preprocess_plain_gen(sources, tokenizer,has_image,info)
    else:
        raise NotImplementedError


import re
import random

def find_brackets(x):
    return re.compile('\[[^\]]+\]').findall(x)


def remove_prefix(x):
    # prefix = ['an image of','a photo of','a painting of','']
    # for p in prefix:
    #     x = x.replace(p,'')
    return x
from instructany2pix.llm.constants import DEFAULT_VIDEO_TOKEN,REPLACEMENT_TYPE
import numpy as np
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        
        with open(data_args.media_map) as f:
            self.TXT2TENSOR = json.loads(f.read())

                  
    def get_tensors_from_str(self,x):
        x = x.replace('[','',).replace(']','',)
        if x not in  self.TXT2TENSOR:
            print(x)
            return torch.zeros((1,1024))
        assert x in self.TXT2TENSOR,repr(x)
        z = self.TXT2TENSOR[x]
        npz_path = os.path.join(self.data_args.image_folder,z['fpath'])
        data = np.load(npz_path)
        assert str(z['key']) in data,data.keys()
        res = torch.tensor(data[str(z['key'])]).view(1,1024)
        res = res / (res.norm()+1e-9) * 20
        return res


    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        inputs = None
        extra_inputs = []
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image_path = os.path.join(image_folder, image_file)
            inputs = {
                "image": [image_path,]
            }
            processor = self.data_args.image_processor
            inputs = processor(inputs)
            inputs['image']['pixel_values'] = inputs['image']['pixel_values'].squeeze(0) # hack
            sources_p = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args) # move <image> to the start of all data
        elif 'multimodal_input' in sources[0]:
            multi_input = sources[0]['multimodal_input']
            modality = multi_input['type']
            assert modality!= 'image'
            if modality == 'audio':
                inputs =  self.data_args.vae_processor(os.path.join(self.data_args.image_folder,multi_input['src']),multi_input['type'])
                extra_inputs.append(
                    {'type': 'audio',
                    'data':inputs}
                )
            sources_p = copy.deepcopy([e["conversations"] for e in sources])
            sources_p[0][0]['value'] = sources_p[0][0]['value'].replace(DEFAULT_AUDIO_TOKEN,DEFAULT_AUDIO_TOKEN*8)
        else:
            sources_p = copy.deepcopy([e["conversations"] for e in sources])
        do_generation=False
        info = {}
        image_folder = self.data_args.image_folder
        if sources[0].get('task') == 'generation':
            do_generation = True
            processor = self.data_args.vae_processor
            generation_target = copy.deepcopy(sources[0]['generation_target'])
            generation_target["data"] = processor(os.path.join(image_folder,generation_target['src']),generation_target['type'])
            info['generation'] = True
            info['generation_type'] = generation_target['type']
            info['generation_seq_len'] = 1 if generation_target['type'] == 'image' else 8 #32 * 32 if generation_target['type'] == 'image' else 80 # FIXME: do not hardcode
        extra_replacement = []
        extra_replacement_mask = []
        if sources[0].get('task') == 'any2any':
            info['generation_seq_len'] = 1
            replacement = []
            replacement_mask = [] # loss mask
            base = sources[0]['base']
            info['generation'] = True
            drop_base = random.random() < 0.2
            if 'added' not in sources[0]:
                sources[0]['added'] = []
            all_tgts = {x[1]:x for x in (sources[0]['added'] if sources[0]['added'] else [])}
            adds = []
            raw_val = []
            for turn in sources_p[0]:
                src = turn['from']
                val = turn['value']
                if drop_base:
                    val = val.replace('<base>','<base_null>')
                if src == 'human':
                    matches = find_brackets(val)
                    for prompt in matches: # list of str wit '[]'
                        if prompt in all_tgts:
                            set_instance = True
                        else:
                            set_instance = False
                        prompt_clean = prompt[1:-1]
                        if clean(prompt_clean) not in  self.TXT2TENSOR:
                            print(prompt_clean)
                            val = val.replace(prompt,remove_prefix(prompt_clean),1)
                            continue
                        if prompt == base:
                            if drop_base:
                                val = val.replace(prompt,remove_prefix(prompt_clean),1)
                            else:
                                val = val.replace(prompt,DEFAULT_VIDEO_TOKEN,1)
                                replacement.append(prompt_clean)
                                replacement_mask.append(REPLACEMENT_TYPE.INPUT)
                                raw_val.append(prompt)
                                # if set_instance:
                                #     adds.append((all_tgts[prompt][0],prompt_clean))
                        elif random.random() < 0.2:
                            val = val.replace(prompt,remove_prefix(prompt_clean),1)
                        else:
                            val = val.replace(prompt,DEFAULT_VIDEO_TOKEN,1)
                            replacement.append(prompt_clean)
                            replacement_mask.append(REPLACEMENT_TYPE.INPUT)
                            raw_val.append(prompt)
                            if set_instance:
                                adds.append((all_tgts[prompt][0],prompt_clean))
                    raw_val.append(val)
                elif src == 'gpt':
                    matches = find_brackets(val)
                    for prompt in matches: # list of str wit '[]'
                        prompt_clean = prompt[1:-1]
                        seen = 0
                        if prompt == base and (drop_base or prompt_clean not in self.TXT2TENSOR):
                            val = val.replace(prompt,'',1)
                            val = val.replace('<base>','<base_null>')
                        elif prompt == base:
                            val = val.replace(prompt,DEFAULT_VIDEO_TOKEN,1)
                            replacement.append(prompt_clean)
                            replacement_mask.append(REPLACEMENT_TYPE.BASE)
                        else:
                            assert seen == 0, "Only one outout per instructions!!!"
                            seen =1
                            if self.data_args.output_text:
                                val = val.replace(prompt,prompt+DEFAULT_VIDEO_TOKEN,1)
                            else:
                                val = val.replace(prompt,DEFAULT_VIDEO_TOKEN,1)
                            replacement.append(prompt_clean)
                            replacement_mask.append(REPLACEMENT_TYPE.GEN)
                    # if (not adds) and all_tgts:
                    #     raise ValueError(f'{adds},{val},{all_tgts},{raw_val}')
                    if adds:
                        val += 'additions:'
                        for addition_src,addition_caption in adds:
                            val += f'{addition_src}:{DEFAULT_VIDEO_TOKEN}.'
                            replacement.append(addition_caption)
                            replacement_mask.append(REPLACEMENT_TYPE.GEN)
                        # raise ValueError(val)
                        # print("MODIFIED:")
                else:
                    raise NotImplemented
                turn['value']= val
                assert len(replacement_mask) == len(replacement)
            if len(replacement):
                extra_replacement = torch.cat([self.get_tensors_from_str(clean(x)) for x in replacement])
            extra_replacement_mask = replacement_mask


        info['output_text'] = self.data_args.output_text
        data_dict = preprocess(
            sources_p,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]),info=info)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        if do_generation:
            data_dict['generation_target'] = generation_target
        else:
            data_dict['generation_target'] = None
        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = inputs # hack, actually multimodal
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = dict(
                image=dict(
                    pixel_values=torch.zeros(3, crop_size['height'], crop_size['width'])
                )
            )
        data_dict['extra_inputs']=extra_inputs
        data_dict['extra_replacement']=extra_replacement
        data_dict['extra_replacement_mask']=extra_replacement_mask
        assert len(extra_replacement) == len(extra_replacement_mask)
        return data_dict
    
from torch.utils.data import default_collate
def gather_by_key(data):
    gathered_inputs = {}
    info = []
    for idx_b,b in enumerate(data):
        for row in b:
            modality,data = row['type'],row['data']
            if modality not in gathered_inputs:
                gathered_inputs[modality] = []
            gathered_inputs[modality].append(data)
            info.append(dict(bn=idx_b,modality=modality,idx=len(gathered_inputs[modality])-1))
    for modality in gathered_inputs.keys():
        gathered_inputs[modality] = default_collate(gathered_inputs[modality])
    gathered_inputs['info'] = info
    return gathered_inputs

def clean(x):
    x = x.lower().strip()
    x = x.replace('.','')
    return x

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        generation_target=list([x['generation_target'] for x in instances])
        gathered_generation_target= {}
        info = []
        for idx,tgt in enumerate(generation_target):
            if tgt is None:
                continue
            modality = tgt['type']
            data = tgt['data']
            if modality not in gathered_generation_target:
                    gathered_generation_target[modality] = []
            gathered_generation_target[modality].append(data)
            info.append(dict(batch=idx,modality=modality,idx=len(gathered_generation_target[modality])-1))
        for modality in gathered_generation_target.keys():
                gathered_generation_target[modality] = default_collate(gathered_generation_target[modality])
        gathered_generation_target['info'] = info
        batch['generation_target']=gathered_generation_target

        extra_replacement=list([x['extra_replacement'] for x in instances if len(x['extra_replacement'])])
        

        extra_replacement_idx = list([torch.tensor([idx,] * len(instances[idx]['extra_replacement']),dtype=torch.long)
                                     for idx in range(len(instances))
                                     ])
        extra_replacement = torch.cat(extra_replacement) # N D
        extra_replacement_idx = torch.cat(extra_replacement_idx) # N
        assert len(extra_replacement) == len(extra_replacement_idx)
        batch['extra_replacement'] = dict(
            data=extra_replacement,
            idx=extra_replacement_idx,
            mask=torch.cat([torch.tensor(x['extra_replacement_mask'],dtype=torch.long) for x in instances])
        )


        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            gathered_inputs = {}
            info = []
            for image in images:
                assert len(image)==1
                modality,data = list(image.items())[0]
                if modality not in gathered_inputs:
                    gathered_inputs[modality] = []
                gathered_inputs[modality].append(data)
                info.append(dict(modality=modality,idx=len(gathered_inputs[modality])-1))
            for modality in gathered_inputs.keys():
                gathered_inputs[modality] = default_collate(gathered_inputs[modality])
            gathered_inputs['info'] = info
            batch['images']=gathered_inputs
        #     if all(x is not None and x.shape == images[0].shape for x in images):
        #         batch['images'] = torch.stack(images)
        #     else:
        #         batch['images'] = images
        batch['extra_inputs']=gather_by_key([instance['extra_inputs'] for instance in instances])
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    global local_rank
    from diffusers import StableUnCLIPImg2ImgPipeline 
    # pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, low_cpu_mem_usage=False, # variation="fp16",
    # )
    pipe = None
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    
    bnb_model_from_pretrained_args = {}
    if training_args.split_loading:
        ranks_LST = [ [0,1,2,3],
            [4,5,6,7],
            [8,9,10,11],]
        # ranks_LST = [ [0,1,],[2,3],
        #     [4,5,],[6,7],
        #     [8,9],[10,11],]
        bnb_model_from_pretrained_args = dict(low_cpu_mem_usage=True,)
    else:
        ranks_LST = [list(range(20)),]
    for ranks in ranks_LST:
        dist.barrier()
        if local_rank in ranks:
            if training_args.bits in [4, 8]:
                from transformers import BitsAndBytesConfig
                bnb_model_from_pretrained_args.update(dict(
                    device_map={"": training_args.device},
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=training_args.bits == 4,
                        load_in_8bit=training_args.bits == 8,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_use_double_quant=training_args.double_quant,
                        bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
                    )
                ))

            if model_args.vision_tower is not None:
                model_cls = InstructAny2PixLMForCausalLM
                config_cls = InstructAny2PixLMConfig
                if model_args.dev == 'test': # test full size no load
                    cfg = config_cls.from_pretrained(model_args.model_name_or_path)
                    model = model_cls._from_config(cfg)
                elif model_args.dev == 'test2': # test 2 layer
                    cfg = config_cls.from_pretrained(model_args.model_name_or_path)
                    cfg.num_hidden_layers = 2
                    model = model_cls._from_config(cfg)
                else:
                    model = model_cls.from_pretrained(
                        model_args.load or model_args.model_name_or_path,
                        cache_dir=training_args.cache_dir,
                        **bnb_model_from_pretrained_args
                    )
            else:
                model = transformers.LlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
            model.config.use_cache = False

            if model_args.freeze_backbone:
                model.model.requires_grad_(False)

            if training_args.bits in [4, 8]:
                from peft import prepare_model_for_kbit_training
                model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

            if training_args.gradient_checkpointing:
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
           
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,
            )

           
            tokenizer.pad_token = tokenizer.unk_token
            if model_args.version in conversation_lib.conv_templates:
                conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
            else:
                conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

            if model_args.vision_tower is not None:
                model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
            if training_args.lora_enable:
                from peft import LoraConfig, get_peft_model
                lora_config = LoraConfig(
                    r=training_args.lora_r,
                    lora_alpha=training_args.lora_alpha,
                    target_modules=find_all_linear_names(model),
                    modules_to_save=['model.vae_predictor_image','model.vae_predictor_audio','lm_head'],
                    lora_dropout=training_args.lora_dropout,
                    bias=training_args.lora_bias,
                    task_type="CAUSAL_LM",
                )
                if training_args.bits == 16:
                    if training_args.bf16:
                        model.to(torch.bfloat16)
                    if training_args.fp16:
                        model.to(torch.float16)
                rank0_print("Adding LoRA adapters...")
                model = get_peft_model(model, lora_config)

            if model_args.vision_tower is not None:
                if not model_args.load:
                    model.get_model().initialize_vision_modules(
                        model_args=model_args,
                        fsdp=training_args.fsdp
                    )

                vision_tower = model.get_vision_tower()
                vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

                data_args.image_processor = vision_tower.image_processor
                if model_args.vae_image or model_args.vae_audio:
                    data_args.vae_processor = model.get_vae().processor
                data_args.is_multimodal = True

                model.config.image_aspect_ratio = data_args.image_aspect_ratio
                model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

                model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
                if model_args.tune_mm_mlp_adapter:
                    model.requires_grad_(False)
                    # HACK
                    for p in model.get_model().mm_projector.parameters():
                        p.requires_grad = True
                    for p in model.lm_head.parameters():
                        p.requires_grad = False
                    for p in model.get_model().vae_projector_image.parameters():
                        p.requires_grad = True
                    for p in model.get_model().vae_predictor_image.parameters():
                        p.requires_grad =False
                    for p in model.get_input_embeddings().parameters():
                        p.requires_grad = False
                    for p in model.get_output_embeddings().parameters():
                        p.requires_grad = False

                model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
                if training_args.freeze_mm_mlp_adapter:
                    for p in model.get_model().mm_projector.parameters():
                        p.requires_grad = False

                if training_args.bits in [4, 8]:
                    model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

                model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
                training_args.use_im_start_end = model_args.mm_use_im_start_end
                model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
                #model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

            if training_args.bits in [4, 8]:
                from peft.tuners.lora import LoraLayer
                for name, module in model.named_modules():
                    if isinstance(module, LoraLayer):
                        if training_args.bf16:
                            module = module.to(torch.bfloat16)
                    if 'norm' in name:
                        module = module.to(torch.float32)
                    if 'lm_head' in name or 'embed_tokens' in name:
                        if hasattr(module, 'weight'):
                            if training_args.bf16 and module.weight.dtype == torch.float32:
                                module = module.to(torch.bfloat16)
            data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
            # if model_args.load:
            #     model.load_state_dict(torch.load(model_args.load))
            trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    pipe=pipe,
                    **data_module)
        else:
            pass

    if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(os.path.join(training_args.output_dir,'tokenizer'))
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        model.config.save_pretrained(training_args.output_dir)
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir,force_save=True)


if __name__ == "__main__":
    train()
