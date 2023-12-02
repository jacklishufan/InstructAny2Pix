#    Copyright 2023 Haotian Liu
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


from typing import List, Optional, Tuple, Union,Dict
from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast,ModelOutput

from ..any2pix_arch import InstructAny2PixLMMetaModel, InstructAny2PixLMMetaForCausalLM
from instructany2pix.llm.constants import IGNORE_INDEX
from transformers.modeling_outputs import BaseModelOutputWithPast
import logging

logger = logging.getLogger(__name__)

class InstructAny2PixLMConfig(LlamaConfig):
    model_type = "instructany2pix"


class InstructAny2PixLMModel(InstructAny2PixLMMetaModel, LlamaModel):
    config_class = InstructAny2PixLMConfig

    def __init__(self, config: LlamaConfig):
        super(InstructAny2PixLMModel, self).__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        #replacement_mask = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        ) 
        # if replacement_mask is not None and len(attention_mask.shape) == 4:
        #     replacement_mask = replacement_mask[:,None,:] & replacement_mask[:,:,None] # B X N X N
        #     replacement_mask = replacement_mask[:,None] # B X H X N X N
        #     attention_mask[replacement_mask] = 0.0
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
from instructany2pix.llm.constants import REPLACEMENT_TYPE

class InstructAny2PixLMForCausalLM(LlamaForCausalLM, InstructAny2PixLMMetaForCausalLM):
    config_class = InstructAny2PixLMConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = InstructAny2PixLMModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        #self.lm_head_img = nn.Linear(3, config.vocab_size, bias=False) # FIXME: Add config
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[Dict] = None,
        return_dict: Optional[bool] = None,
        generation_target: Optional[Dict] = None,
        return_generations=True,
        extra_inputs=None,
        extra_replacement=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if self.training:
            encodings = self.get_model().get_vae()(generation_target)
        else:
            encodings = {}
        # encode vae for 
        replacement_mask = torch.zeros_like(input_ids,dtype=bool).to(input_ids.device)
        if 'image' in encodings:
            quant_image,ind_image,info_image = encodings['image']
            n = quant_image.shape[0]
            if info_image:
                ind_image = ind_image.view(n,-1)
                loss_fct_img = nn.CrossEntropyLoss()
                img_loss_obj = 'ar'
            else:
                ind_image = quant_image.squeeze(-1).squeeze(-1).unsqueeze(1)
                loss_fct_img = nn.MSELoss()
                img_loss_obj = 'latent'
            img_embded = self.get_model().vae_projector_image(rearrange(quant_image,'n c h w -> n (h w) c') )
            replacement_mask_img = input_ids ==  self.DEFAULT_IM_GEN_TOKEN_IDX
            replacement_mask = replacement_mask | replacement_mask_img
        if 'audio' in encodings:
            quant_audio,ind_audio,info_audio = encodings['audio']
            n = quant_audio.shape[0]
            if info_audio:
                ind_audio = ind_audio.view(n,-1)
                loss_fct_aud = nn.CrossEntropyLoss()
                audloss_obj = 'ar'
            else:
                ind_aud = quant_audio.squeeze(1)# N H W C -> N L C
                loss_fct_aud = nn.MSELoss()
                aud_loss_obj = 'latent'
            audio_embded = self.get_model().vae_projector_audio(rearrange(quant_audio,'n h w c-> n (h w) c') )
            replacement_mask_audio = input_ids ==  self.DEFAULT_AUDIO_GEN_TOKEN_IDX
            replacement_mask = replacement_mask | replacement_mask_audio

        # replacement_audio = input_ids ==  self.DEFAULT_AUDIO_GEN_TOKEN_IDX
        # audio_input_embded = self.get_model().vae_projector_audio(rearrange(quant_audio,'n h w c-> n (h w) c') )
        
        raw_input_ids = input_ids
        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images,novision=True)
        if extra_replacement is not None:
            if self.training:
                extra_replacement_mask = (raw_input_ids == self.DEFAULT_VIDEO_TOKEN_IDX ) # | (
                # raw_input_ids == self.DEFAULT_IM_GEN_TOKEN_IDX) | (raw_input_ids == self.DEFAULT_BASE_TOKEN_IDX) self.get_model().vae_projector_image[0].weight.grad
                if extra_replacement['mask'].shape[0] != inputs_embeds[extra_replacement_mask].shape[0]:
                    print("SKIPPED")
                    extra_replacement['mask'] = torch.zeros(inputs_embeds[extra_replacement_mask].shape[0]).to(extra_replacement['mask'])     
                z = torch.zeros_like(inputs_embeds)
                z2 = self.get_model().vae_projector_image(
                    extra_replacement['data'][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT])
                a,b = torch.where(extra_replacement_mask)
                z[a[extra_replacement['mask']==REPLACEMENT_TYPE.INPUT],b[extra_replacement['mask']==REPLACEMENT_TYPE.INPUT]] += z2
                inputs_embeds[extra_replacement_mask][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT] = 0.0
                z = z + inputs_embeds
                inputs_embeds = z
                print("Replaced:",len(extra_replacement['mask']==REPLACEMENT_TYPE.INPUT))
                extra_tgt_mask = (extra_replacement['mask']==REPLACEMENT_TYPE.BASE )| (extra_replacement['mask']==REPLACEMENT_TYPE.GEN)
                extra_replacement_gt = extra_replacement['data'][extra_tgt_mask]
                loss_fn_extra = nn.L1Loss()
                if extra_replacement_gt.shape[0]==0:
                    loss_fn_extra = None
            else:
                assert labels is None
                extra_replacement_mask = (raw_input_ids == self.DEFAULT_VIDEO_TOKEN_IDX )
                #print(len(extra_replacement['mask']==REPLACEMENT_TYPE.INPUT))

                z = torch.zeros_like(inputs_embeds)
                z2 = self.get_model().vae_projector_image(
                    extra_replacement['data'][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT])
                #print("z2",z2)
                a,b = torch.where(extra_replacement_mask)
                a = a[:extra_replacement['mask'].shape[0]]
                b = b[:extra_replacement['mask'].shape[0]]
                z[a[extra_replacement['mask']==REPLACEMENT_TYPE.INPUT],b[extra_replacement['mask']==REPLACEMENT_TYPE.INPUT]] += z2
                inputs_embeds[extra_replacement_mask][:extra_replacement['mask'].shape[0]][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT] = 0.0
                z = z + inputs_embeds
                inputs_embeds = z
                #print("HERE")
                #print(inputs_embeds[extra_replacement_mask][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT])
                
                # inputs_embeds[extra_replacement_mask][:extra_replacement['mask'].shape[0]][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT] = self.get_model().vae_projector_image(
                #     extra_replacement['data'][extra_replacement['mask']==REPLACEMENT_TYPE.INPUT].to(inputs_embeds))
            # z.sum().backward()
        if self.training:
            for replace_info in generation_target['info']:
                ii = replace_info['idx']
                mm = replace_info['modality']
                if mm == 'image':
                    inputs_embeds[replace_info['batch']][replacement_mask_img[replace_info['batch']]] = img_embded[ii]
                    #labels[replace_info['batch']][torch.where(replacement_mask_img[ii])[0]-1] just for sanity check <gen start>, <gen> ....
                elif mm == 'audio':
                    inputs_embeds[replace_info['batch']][replacement_mask_audio[replace_info['batch']]] = audio_embded[ii]
        # labels[labels==self.DEFAULT_IM_GEN_TOKEN_IDX] = IGNORE_INDEX
        # labels[labels==self.DEFAULT_AUDIO_GEN_TOKEN_IDX] = IGNORE_INDEX
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        
        if extra_inputs is not None:
            if 'audio' in extra_inputs:
                audio_in = extra_inputs['audio'].squeeze(1) # N 8 768
                audio_in = self.get_model().vae_projector_audio(audio_in)
                msk_in = raw_input_ids ==  self.DEFAULT_AUDIO_TOKEN_IDX
                for rinfo in extra_inputs['info']:
                    l_mask = msk_in[rinfo['bn']]
                    if l_mask.sum() !=8:
                        continue
                    else:
                        inputs_embeds[rinfo['bn'],l_mask]=audio_in[rinfo['idx']]
                # hack, assume has N audio inp
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
           # replacement_mask=replacement_mask, # allow looking into future for images
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        img_decode = None
        aud_decode=None
        individual_losses = {}
        extra_gen = None
        extra_gen_idx = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            hidden_states.shape
            output_vae_img = []
            target_vae_img = []
            output_vae_audio = []
            target_vae_audio = []
            for replace_info in generation_target['info']:
                ii = replace_info['idx']
                mm = replace_info['modality']
                if mm == 'image':
                    output_vae_img.append(hidden_states[replace_info['batch']][:-1][replacement_mask_img[replace_info['batch']][1:]])
                    target_vae_img.append(ind_image[ii])
                    #labels[replace_info['batch']][torch.where(replacement_mask_img[ii])[0]-1] just for sanity check <gen start>, <gen> ....
                elif mm == 'audio':
                    output_vae_audio.append(hidden_states[replace_info['batch']][:-1][replacement_mask_audio[replace_info['batch']][1:]])
                    target_vae_audio.append(ind_aud[ii])
            loss_fct = CrossEntropyLoss()

                #prediction = logits_img.argmax(-1)
     
            # Flatten the tokens
            
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss_lang = loss.detach().item()
            individual_losses['loss_lang'] = loss_lang
            if len(output_vae_img):
                logits_img = self.get_model().vae_predictor_image(torch.cat(output_vae_img))
                tgt_img = torch.cat(target_vae_img)
                if img_loss_obj =='ar':
                    tgt_img = tgt_img.view(-1) # discrete tokens
                loss_img = loss_fct_img(logits_img,tgt_img)
                if img_loss_obj =='ar':
                    pass
                else:
                    loss_img *= logits_img.shape[-1]
                loss += loss_img
                individual_losses['loss_img'] =loss_img.detach().item()
                if return_generations:
                    with torch.no_grad():
                        if img_loss_obj =='ar':
                            img_encodings_pred = logits_img.argmax(-1).view(len(output_vae_img),-1)
                        else:
                            img_encodings_pred = logits_img.detach() # N, D_emb
                        img_decode = self.get_vae().image_vae.decode_seq(img_encodings_pred,info_image)
            
            if len(output_vae_audio):
                logits_aud = self.get_model().vae_predictor_audio(torch.cat(output_vae_audio))
                tgt_aud = torch.cat(target_vae_audio)
                if aud_loss_obj =='ar':
                    tgt_aud = tgt_aud.view(-1) # discrete tokens
                loss_aud = loss_fct_aud(logits_aud,tgt_aud)
                if aud_loss_obj =='ar':
                    pass
                else:
                    loss_aud *= logits_aud.shape[-1]
                loss += loss_aud
                individual_losses['loss_aud'] =loss_aud.detach().item()
                if return_generations:
                    with torch.no_grad():
                        if aud_loss_obj =='ar':
                            aud_encodings_pred = logits_aud.argmax(-1).view(len(output_vae_audio),-1)
                        else:
                            aud_encodings_pred = logits_aud.detach() # N, D_emb
                        aud_decode = self.get_vae().image_vae.decode_seq(aud_encodings_pred,info_audio)
            if extra_replacement is not None:
                extra_pred = self.get_model().vae_predictor_image(hidden_states[:,:-1][extra_replacement_mask[:,1:]])
                extra_pred = extra_pred[extra_tgt_mask]
                if labels is not None:
                    if loss_fn_extra is None:
                        loss_extra = extra_pred.sum() * 0.0
                    else:
                        loss_extra = loss_fn_extra(extra_pred,extra_replacement_gt)
                    if torch.isnan(loss_extra):
                        loss_extra = 0.0
                    loss += loss_extra
                    individual_losses['loss_extra'] =loss_extra.detach().item()
                if return_generations:
                    extra_gen = extra_pred
                    extra_gen_idx = extra_replacement_mask[:,1:]
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ModelOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            extra_gen=extra_gen,
            extra_gen_idx=extra_gen_idx,
            attentions=outputs.attentions,
            img_decode=img_decode,
            aud_decode=aud_decode,
            individual_losses=individual_losses,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "extra_replacement": kwargs.get("extra_replacement", None),
            }
        )
        return model_inputs

AutoConfig.register("instructany2pix", InstructAny2PixLMConfig)
AutoModelForCausalLM.register(InstructAny2PixLMConfig, InstructAny2PixLMForCausalLM)
