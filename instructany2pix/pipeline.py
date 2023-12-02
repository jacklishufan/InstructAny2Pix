from typing import Any
import torch
import os
from .prior import InstructAny2PixPrior,prior_config

from imagebind import data as image_bind_data
from imagebind import imagebind_model
from .llm.model import InstructAny2PixLMForCausalLM
from .prior.model import MODALITY
import transformers
import torch
from instructany2pix.llm.mm_utils import KeywordsStoppingCriteria
from .diffusion.sdxl_img2img_pipeline import build_sdxl
from instructany2pix.llm.conversation import conv_templates, SeparatorStyle
import torchaudio
from diffusers import StableDiffusionXLImg2ImgPipeline
def build_lm(ckpt='ckpts/llm'):
    any2pix_tokenizer = transformers.AutoTokenizer.from_pretrained(
        ckpt,subfolder='tokenizer'
    )

    any2pix_lm = None
    any2pix_lm = InstructAny2PixLMForCausalLM.from_pretrained(
        ckpt,
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float32,
                            # config=config
                                #cache_dir=training_args.cache_dir,
                                #**bnb_model_from_pretrained_args
                            )

    any2pix_lm.initialize_vision_tokenizer(None,any2pix_tokenizer,True)
    return any2pix_tokenizer,any2pix_lm
from .ddim.pnp_pipeline import SDXLDDIMPipeline
from diffusers import DDIMScheduler
from PIL import Image
def resize_and_crop(img, size, crop_type='middle'):
    """
    Resize and crop an image to fit the specified size.
    args:
        img_path: path for the image to resize.
        modified_path: path to store the modified image.
        size: `(width, height)` tuple.
        crop_type: can be 'top', 'middle' or 'bottom', depending on this
            value, the image will cropped getting the 'top/left', 'midle' or
            'bottom/rigth' of the image to fit the size.
    raises:
        Exception: if can not open the file in img_path of there is problems
            to save the image.
        ValueError: if an invalid `crop_type` is provided.
    """
    # If height is higher we resize vertically, if not we resize horizontally
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    #The image is scaled/cropped vertically or horizontally depending on the ratio
    if ratio > img_ratio:
        img = img.resize((size[0], int(size[0] * img.size[1] / img.size[0])),)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, img.size[0], size[1])
        elif crop_type == 'middle':
            box = (0, (img.size[1] - size[1]) / 2, img.size[0], (img.size[1] + size[1]) / 2)
        elif crop_type == 'bottom':
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else :
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize((int(size[1] * img.size[0] / img.size[1]), size[1]),
                )
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, size[0], img.size[1])
        elif crop_type == 'middle':
            box = ((img.size[0] - size[0]) / 2, 0, (img.size[0] + size[0]) / 2, img.size[1])
        elif crop_type == 'bottom':
            box = (img.size[0] - size[0], 0, img.size[0], img.size[1])
        else :
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    else :
        img = img.resize((int(size[0]), int(size[1])),)
    return img
class REPLACEMENT_TYPE:
            INPUT = 0
            BASE = 1
            GEN = 2

class InstructAny2PixPipeline:

    def __init__(self,ckpt='ckpts') -> None:
        model = InstructAny2PixPrior(**prior_config)
        model = model.eval()
        # model.load_state_dict(torch.load('../diffusion_prior.bin',map_location='cpu'))
        pipe = build_sdxl(ckpt=os.path.join(ckpt,'sdxl'))

        new_sch = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe_inversion = SDXLDDIMPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                text_encoder_2=pipe.text_encoder_2,
                unet=pipe.unet,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
        )
        pipe_inversion.scheduler = new_sch
        any2pix_tokenizer,any2pix_lm = build_lm(os.path.join(ckpt,'llm'))
        model_imb = imagebind_model.imagebind_huge(pretrained=False)
        model_imb.load_state_dict(torch.load(os.path.join(ckpt,'imagebind_huge.pth'),map_location='cpu'))
        model.load_state_dict(torch.load(os.path.join(ckpt,'prior/model.bin'),map_location='cpu'))
        model_imb = model_imb.eval()
        self.model = model
        self.pipe_inversion = pipe_inversion
        self.pipe = pipe.to('cuda').to(torch.float16)
        self.model_imb = model_imb
        #self.audioldm_dict = audioldm_dict
        self.any2pix_tokenizer = any2pix_tokenizer
        self.any2pix_lm = any2pix_lm
        self.piperf = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        self.piperf = self.piperf.to("cuda").to(torch.float16)
        self.cache = None

    def forward_llm(self,inst,mm_data=[],use_cache=False):
        if use_cache:
            return self.cache
        all_tensors = []
        for r in mm_data:
            dict_type = r['type']
            fpath = r['fname']
            inputs = {}
            if dict_type == 'audio':
                inputs[dict_type] = image_bind_data.load_and_transform_audio_data([fpath],'cpu')
                res = self.model_imb(inputs)['audio']
            elif dict_type == 'image':
                inputs['vision'] = image_bind_data.load_and_transform_vision_data([fpath],'cpu')
                res = self.model_imb(inputs)['vision']
            
            all_tensors.append(res)
        aux_info = torch.cat(all_tensors)
        aux_info = aux_info / (aux_info.norm(dim=-1,keepdim=True)+1e-9) * 20
        

        extra_replacement = {
            "data":aux_info,
            "mask":torch.tensor([REPLACEMENT_TYPE.INPUT]*aux_info.shape[0],dtype=torch.long)
        }

        conv = conv_templates['vicuna_v1'].copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conv.append_message(conv.roles[0],inst)
        #conv.append_message(conv.roles[0], inst)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = self.any2pix_tokenizer(prompt, return_tensors='pt').input_ids
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.any2pix_tokenizer, input_ids)

        gen_seq = self.any2pix_tokenizer('<video>',add_special_tokens=False)
        gen_seq = gen_seq.input_ids[0]
        gen_seq
        base_null = self.any2pix_tokenizer('<base_null>',add_special_tokens=False)
        base_null = base_null.input_ids[0]
        base_null
        base_tkn = self.any2pix_tokenizer('<base>',add_special_tokens=False)
        base_tkn = base_tkn.input_ids[0]
        base_tkn

        self.any2pix_lm.eval()

        output_ids = self.any2pix_lm.generate(
            input_ids.cuda(),
            images=None,
            do_sample=True,
            temperature=0.3,
            max_new_tokens=100,
            output_hidden_states=True,
            use_cache=False,
            return_dict_in_generate=True,
            extra_replacement=extra_replacement,
            stopping_criteria=[stopping_criteria])

        out_seq = output_ids.sequences[:,input_ids.shape[1]:]
        assert len(output_ids.hidden_states) == out_seq.shape[1]
        gen_idx = torch.where(out_seq.view(-1) == gen_seq)[0][-1].item()
        with torch.no_grad():
            image_embeds = self.any2pix_lm.get_model().vae_predictor_image(output_ids.hidden_states[-2][-1][:,-1:]).detach().cpu()
            #image_embeds = image_embeds / image_embeds.norm() * 22

        gen_idx = out_seq.view(-1).tolist().index(base_tkn)+1
        with torch.no_grad():
            base_embed = self.any2pix_lm.get_model().vae_predictor_image(output_ids.hidden_states[gen_idx][-1][:,-1:]).detach().cpu()
        base_embed = base_embed[0]
        base_idx = torch.einsum('ac,bc->ab',base_embed.float().detach().cpu() / base_embed.norm() * 20,aux_info.float() )[0].argmax().item()
        b = mm_data[base_idx]['fname']
        #assert a == 'image'
        tp = self.any2pix_tokenizer.batch_decode(output_ids.sequences)
        import re
        output_caption = re.compile('\[([^\]]+)\]').findall(tp[0])[0]
        return image_embeds,base_embed,output_caption,b
    
    def loas_base_img(self,base_img_path):
        img_base = Image.open(base_img_path)
        img_base = resize_and_crop(img_base,(1024,1024),crop_type='middle')
        img_base = img_base.resize((1024,1024))
        return img_base
    
    def polar_intrtpolate(self,x,y,alpha):
        n0 = x.norm()
        n1 = y.norm()
        ll = x * alpha + y * (1-alpha)
        n = n0 * alpha + n1 * (1-alpha)
        return ll / ll.norm() * n
    
    def __call__(self, inst,mm_data,alpha = 0.7,h=[0.0,0.4,1.0],norm=20.0,refinement=0.5,llm_only=False,num_inference_steps=25,use_cache=False,debug=False) -> Any:
        image_embeds,base_embed,output_caption,base_img_path = self.forward_llm(inst,mm_data,use_cache=use_cache)
        self.cache = image_embeds,base_embed,output_caption,base_img_path
        if llm_only:
            return None,None,output_caption
        y = self.model.generate_diffusion(MODALITY.VIDEO,MODALITY.IMAGE,image_embeds / image_embeds.norm() * 100,device='cpu',
                             no_diffusion=True,num_inference_steps=25,
                             image_bind_overwrite=None,
                             dtype=torch.float32,guidance_scale=10,
                             force_guidence_t0=True,do_classifier_free_guidance=True,score=6.5)
        # y = self.model.generate_diffusion(MODALITY.TEXT,MODALITY.IMAGE,[output_caption],device='cpu',
        #                      no_diffusion=True,num_inference_steps=25,
        #                      image_bind_overwrite=None,
        #                      dtype=torch.float32,guidance_scale=10,
        #                      force_guidence_t0=True,do_classifier_free_guidance=True,score=6.5)
        
        img_base = self.loas_base_img(base_img_path)

        
        latent_la = base_embed * h[0] + image_embeds * h[1] +y[0] / y[0].norm() *20.0 *h[2] #+y[0] #/y[0].norm()*20# + image_embeds/image_embeds.norm()*10
        latent_la = latent_la.detach().clone()
        latent_la = latent_la / latent_la.norm() * norm
        null_prompt_embeds = torch.zeros(1,768)
        extra_kwargs = {}

        null_prompt_embeds = torch.zeros(1,768)
        latent_inv = self.pipe_inversion.inverse(num_inference_steps=num_inference_steps,
                                            prompt_embeds=torch.zeros(1,1,1024),pooled_prompt_embeds=null_prompt_embeds,image=img_base.resize((1024,1024)))
        latent_inv = latent_inv.images.cpu()
        #alpha = 0.7
        latent_inv = self.polar_intrtpolate(
            latent_inv,
            torch.randn_like(latent_inv),
            alpha,
        )
        extra_kwargs['latents']= latent_inv # latent_inv
        # pipe = pipe.to('cuda').to(torch.float16)
        images = self.pipe(original_size=(1024,1024),
        height=1024,
        width=1024,
        prompt_embeds=latent_la.to(torch.bfloat16).cuda(),
        pooled_prompt_embeds=null_prompt_embeds.to(torch.bfloat16).cuda(), generator=None,
            num_inference_steps=num_inference_steps,guidance_scale=10,**extra_kwargs)
        non_refined =  images[0][0]
        if refinement > 0:
            oo = self.piperf(image=images[0][0],prompt=output_caption+',high quality,well-formed,award-winning',strength=refinement,).images[0]
        else:
            oo = images[0][0]
        if not debug:
            msg = "SUCCESS!"
        else:
            msg = output_caption
        return non_refined,oo,msg
    
def load_json(fp):
    with open(fp) as f:
        data = json.loads(f.read())

    return data

def preprocess_mm_data(z,img_folder,audio_folder):
    new_data = []
    for x in z:
        payload = {k:v for k,v in x.items()}
        if payload['type'] == 'image':
            payload['fname'] = os.path.join(img_folder,payload['fname'])
        elif payload['type'] == 'audio':
            payload['fname'] = os.path.join(audio_folder,payload['fname'])
        new_data.append(payload)
    return new_data

from pathlib import Path
import os
import shutil
import json
def dump_json(data,fp):
    with open(fp,'w') as f:
        f.write(json.dumps(data))
    


