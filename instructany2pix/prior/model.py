import torch
import torch.nn as nn
from ..utils import (
    instantiate_from_config,
)
class MODALITY:
    IMAGE = 0
    AUDIO = 1 
    TEXT = 2
    VIDEO = 3
# from latent_diffusion.modules.encoders.modules import CLAPAudioEmbeddingClassifierFreev2
from transformers import GPT2Config, GPT2Model,AutoTokenizer,CLIPTextModel
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import AdamW
from diffusers import DDPMScheduler
from diffusers.models.embeddings import get_timestep_embedding
import inspect
from tqdm.cli import tqdm
import logging
class CLIPTextModelHiddenState(nn.Module):
    """
    llama = FlanT5HiddenState()
    data = ["","this is not an empty sentence"]
    encoder_hidden_states = llama(data)
    import ipdb;ipdb.set_trace()
    """

    def __init__(
        self, text_encoder_name='laion/CLIP-ViT-H-14-laion2B-s32B-b79K', freeze_text_encoder=True
    ):
        super().__init__()
        self.freeze_text_encoder = freeze_text_encoder
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.model = CLIPTextModel.from_pretrained(text_encoder_name)
        if freeze_text_encoder:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
        else:
            print("=> The text encoder is learnable")

        self.empty_hidden_state_cfg = None
        self.device = None

    # Required
    def get_unconditional_condition(self, batchsize):
        param = next(self.model.parameters())
        if self.freeze_text_encoder:
            assert param.requires_grad == False

        # device = param.device
        if self.empty_hidden_state_cfg is None:
            self.empty_hidden_state_cfg, _ = self([""])

        hidden_state = torch.cat([self.empty_hidden_state_cfg] * batchsize).float()
        attention_mask = (
            torch.ones((batchsize, hidden_state.size(1)))
            .to(hidden_state.device)
            .float()
        )
        return [hidden_state, attention_mask]  # Need to return float type

    def forward(self, batch):
        param = next(self.model.parameters())
        if self.freeze_text_encoder:
            assert param.requires_grad == False

        if self.device is None:
            self.device = param.device

        # print("Manually change text")
        # for i in range(len(batch)):
        #     batch[i] = "dog barking"
        try:
            return self.encode_text(batch)
        except Exception as e:
            print(e, batch)
            logging.exception("An error occurred: %s", str(e))

    def encode_text(self, prompt):
        device = self.model.device
        batch = self.tokenizer(
            prompt,
            max_length=77,  # self.tokenizer.model_max_length
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(
            device
        )
        # Get text encoding
        if self.freeze_text_encoder:
            with torch.no_grad():
                encoder_hidden_states = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )[0]
        else:
            encoder_hidden_states = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )[0]
        return [
            encoder_hidden_states.detach(),
            attention_mask.float(),
        ]  # Attention mask == 1 means usable token



class InstructAny2PixPrior(nn.Module):
    def __init__(
        self,
        base_learning_rate,
        sequence_gen_length,
        sequence_input_key,
        sequence_input_embed_dim,
        cond_stage_config,
        optimizer_type="AdamW",
        use_warmup=True,
        use_ar_gen_loss=False,
        use_audiomae_linear=False,
        target_tokens_mask_ratio=0.0,
        random_mask_ratio=False,
        loss='L1',
        pretrained_name='gpt2',
        embed_dim=768,
        output_dim=None,
        weight_decay=0.,
        diffusion=None,
        **kwargs
    ):
        super().__init__()
        self.diffusion = diffusion
        if diffusion:
            self.noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
            #self.noise_scheduler. sconfig.prediction_type == "sample"
        assert use_audiomae_linear == False
        self.random_mask_ratio = random_mask_ratio
        self.learning_rate = base_learning_rate
        self.cond_stage_config = cond_stage_config
        self.use_audiomae_linear = use_audiomae_linear
        self.optimizer_type = optimizer_type
        self.use_warmup = use_warmup
        self.use_ar_gen_loss = use_ar_gen_loss
        self.weight_decay = weight_decay
        # Even though the LDM can be conditioned on mutliple pooling rate
        # Our model always predict the higest pooling rate

        # self.time_pool = max(self.cond_stage_config["crossattn_audiomae_pooled"]["params"]["time_pooling_factors"])
        # self.freq_pool = max(self.cond_stage_config["crossattn_audiomae_pooled"]["params"]["freq_pooling_factors"])
        # self.mae_token_num = int(512/(self.time_pool*self.freq_pool))

        self.mae_token_num = sequence_gen_length
        self.sequence_input_key = sequence_input_key
        self.sequence_input_embed_dim = sequence_input_embed_dim
        self.target_tokens_mask_ratio = target_tokens_mask_ratio

        self.start_of_sequence_tokens = nn.Embedding(32, embed_dim)
        self.end_of_sequence_tokens = nn.Embedding(32, embed_dim)

        self.input_sequence_embed_linear = nn.ModuleList([])
        self.initial_learning_rate = None

        for dim in self.sequence_input_embed_dim:
            if dim == 0:
                self.input_sequence_embed_linear.append(nn.Identity())
            else: 
                self.input_sequence_embed_linear.append(nn.Linear(dim, embed_dim))

        self.modality_embedding = nn.Embedding(10,embed_dim)

        if output_dim is not None and output_dim != embed_dim:
            self.output_proj = nn.Linear(embed_dim,output_dim)
        else:
            self.output_proj = nn.Identity()

        self.cond_stage_models = nn.ModuleList([])
        self.instantiate_cond_stage(cond_stage_config)
        self.initialize_param_check_toolkit()

        # configuration = GPT2Config(n_layer=1) # TODO
        # self.model=GPT2Model(configuration)
        ###################
        # self.model=nn.Linear(768,768, bias=False) # TODO change the model
        # with torch.no_grad():
        #     self.model.weight.copy_(torch.eye(768))
        ###################
        self.model = GPT2Model(GPT2Config.from_pretrained(pretrained_name))
        ###################
        # self.model = nn.LSTM(input_size=768, hidden_size=768, num_layers=1,bias=False) # TODO

        # self.loss_fn = nn.MSELoss()
        if loss == 'L1':   
            self.loss_fn = nn.L1Loss()
        elif loss == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            raise NotImplemented
        

        self.embed_dim = embed_dim

        self.logger_save_dir = None
        self.logger_exp_name = None
        self.logger_exp_group_name = None
        self.logger_version = None

    
    def get_eps(self,timestep,sample,model_output):
        #print(sample.shape,model_output.shape)
        t = timestep
        prev_timestep = timestep - self.noise_scheduler.config.num_train_timesteps // self.noise_scheduler.num_inference_steps

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            raise NotImplemented
        else:
            predicted_variance = None

        prev_timestep = timestep - self.noise_scheduler.config.num_train_timesteps // self.noise_scheduler.num_inference_steps
        # 1. compute alphas, betas
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.noise_scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.noise_scheduler.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        # current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        # current_beta_t = 1 - current_alpha_t
        # alpha_prod_t = self.noise_scheduler.alphas_cumprod[timestep]
        # alpha_prod_t_prev = self.noise_scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.noise_scheduler.final_alpha_cumprod

        #beta_prod_t = 1 - alpha_prod_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        # if self.config.prediction_type == "epsilon":
        #     pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        # solve for model_output
        #      (sample - alpha_prod_t ** (0.5) * pred_original_sample ) / beta_prod_t ** (0.5) = model_output 
        eps =  (sample - alpha_prod_t ** (0.5) * model_output ) / beta_prod_t ** (0.5) 
        #print(eps.mean(),eps.std())
        return eps


    def set_log_dir(self, save_dir, exp_group_name, exp_name):
        self.logger_save_dir = save_dir
        self.logger_exp_group_name = exp_group_name
        self.logger_exp_name = exp_name

    def cfg_uncond(self, batch_size):
        unconditional_conditioning = {}
        for key in self.cond_stage_model_metadata:
            model_idx = self.cond_stage_model_metadata[key]["model_idx"]
            unconditional_conditioning[key] = self.cond_stage_models[
                model_idx
            ].get_unconditional_condition(batch_size)
        assert (
            "crossattn_audiomae_pooled" in unconditional_conditioning.keys()
        ), "The module is not initialized with AudioMAE"
        unconditional_conditioning[
            "crossattn_clap_to_audiomae_feature"
        ] = unconditional_conditioning["crossattn_audiomae_pooled"]
        return unconditional_conditioning

    def configure_optimizers(self,step_size=5):
        lr = float(self.learning_rate)
        # params = list(self.model.parameters()) + list(self.input_sequence_embed_linear.parameters())
        params = list(self.parameters())

        # opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.98), eps=1e-9)
        opt = eval(self.optimizer_type)(params, lr=lr,weight_decay=self.weight_decay)
        scheduler = lr_scheduler.StepLR(opt, step_size=step_size, gamma=0.8)
        return [opt], [scheduler]

    def add_sos_eos_tokens(self, _id, sequence, attn_mask):
        batchsize = sequence.size(0)

        new_attn_mask_step = torch.ones((batchsize, 1)).to(sequence.device)
        key_id = torch.tensor([_id]).to(sequence.device)

        # Add two more steps to attn mask
        new_attn_mask = torch.cat(
            [new_attn_mask_step, attn_mask, new_attn_mask_step], dim=1
        )

        # Add two more tokens in the sequence
        sos_token = self.start_of_sequence_tokens(key_id).expand(batchsize, 1, -1)
        eos_token = self.end_of_sequence_tokens(key_id).expand(batchsize, 1, -1)
        new_sequence = torch.cat([sos_token, sequence, eos_token], dim=1)
        return new_sequence, new_attn_mask

    def truncate_sequence_and_mask(self, sequence, mask, max_len=512):
        if sequence.size(1) > max_len:
            print(
                "The input sequence length to GPT-2 model is too long:",
                sequence.size(1),
            )
            return sequence[:, :max_len], mask[:, :max_len]
        else:
            return sequence, mask

    def get_input_sequence_and_mask(self, cond_dict):
        input_embeds = None
        input_embeds_attn_mask = None
        for _id, sequence_key in enumerate(self.sequence_input_key):
            # assert sequence_key in cond_dict.keys(), (
            #     "Invalid sequence key %s" % sequence_key
            # )
            if sequence_key not in cond_dict.keys():
                continue
            cond_embed = cond_dict[sequence_key]
            if sequence_key in ['src_type','tgt_type']:
                if len(cond_embed.shape) == 1:
                    cond_embed = cond_embed[:,None]
                item_input_embeds = self.modality_embedding(cond_embed)
                item_attn_mask = torch.ones((cond_embed.size(0), cond_embed.size(1))).to(
                    cond_embed.device
                )
                if input_embeds is None and input_embeds_attn_mask is None:
                    input_embeds, input_embeds_attn_mask = (
                        item_input_embeds,
                        item_attn_mask,
                    )
                else:
                    input_embeds, input_embeds_attn_mask = torch.cat(
                        [input_embeds, item_input_embeds], dim=1
                    ), torch.cat([input_embeds_attn_mask, item_attn_mask], dim=1)
            elif isinstance(cond_embed, list):
                assert (
                    len(cond_embed) == 2
                ), "The crossattn returned list should have length 2, including embed and attn_mask"
                item_input_embeds, item_attn_mask = cond_embed

                item_input_embeds = self.input_sequence_embed_linear[_id](
                    item_input_embeds
                )

                item_input_embeds, item_attn_mask = self.add_sos_eos_tokens(
                    _id, item_input_embeds, item_attn_mask
                )

                if input_embeds is None and input_embeds_attn_mask is None:
                    input_embeds, input_embeds_attn_mask = (
                        item_input_embeds,
                        item_attn_mask,
                    )
                else:
                    input_embeds = torch.cat(
                        [input_embeds, item_input_embeds], dim=1
                    )  # The 1-st dimension is time steps
                    input_embeds_attn_mask = torch.cat(
                        [input_embeds_attn_mask, item_attn_mask], dim=1
                    )  # The 1-st dimension is time steps
            else:
                assert isinstance(cond_embed, torch.Tensor)
                cond_embed = self.input_sequence_embed_linear[_id](cond_embed)
                attn_mask = torch.ones((cond_embed.size(0), cond_embed.size(1))).to(
                    cond_embed.device
                )

                item_input_embeds, item_attn_mask = self.add_sos_eos_tokens(
                    _id, cond_embed, attn_mask
                )

                if input_embeds is None and input_embeds_attn_mask is None:
                    input_embeds, input_embeds_attn_mask = (
                        item_input_embeds,
                        item_attn_mask,
                    )
                else:
                    input_embeds, input_embeds_attn_mask = torch.cat(
                        [input_embeds, item_input_embeds], dim=1
                    ), torch.cat([input_embeds_attn_mask, item_attn_mask], dim=1)

        assert input_embeds is not None and input_embeds_attn_mask is not None

        input_embeds, input_embeds_attn_mask = self.truncate_sequence_and_mask(
            input_embeds, input_embeds_attn_mask, int(1024 - self.mae_token_num)
        )
        cond_sequence_end_time_idx = input_embeds.size(
            1
        )  # The index that we start to collect the output embeds

        return input_embeds, input_embeds_attn_mask, cond_sequence_end_time_idx

    def warmup_step(self):
        if self.initial_learning_rate is None:
            self.initial_learning_rate = float(self.learning_rate)

        # Only the first parameter group
        if self.global_step <= 1000:
            if self.global_step == 0:
                print(
                    "Warming up learning rate start with %s"
                    % self.initial_learning_rate
                )
            self.trainer.optimizers[0].param_groups[0]["lr"] = (
                self.global_step / 1000
            ) * self.initial_learning_rate
        else:
            # TODO set learning rate here
            self.trainer.optimizers[0].param_groups[0][
                "lr"
            ] = self.initial_learning_rate

    def mask_target_sequence(self, target_embeds, target_embeds_attn_mask):
        time_seq_mask = None
        if self.target_tokens_mask_ratio > 1e-4:
            batchsize, time_seq_len, embed_dim = target_embeds.size()
            _, time_seq_len = target_embeds_attn_mask.size()
            # Generate random mask
            if self.random_mask_ratio:
                mask_ratio = torch.rand(1).item() * self.target_tokens_mask_ratio
            else:
                mask_ratio = self.target_tokens_mask_ratio

            time_seq_mask = (torch.rand((batchsize, time_seq_len)) > mask_ratio).to(
                target_embeds.device
            )
            # Mask the target embedding
            target_embeds = target_embeds * time_seq_mask.unsqueeze(-1)
            target_embeds_attn_mask = target_embeds_attn_mask * time_seq_mask
        return target_embeds, target_embeds_attn_mask, time_seq_mask

    def generate_partial(self, batch, cond_dict=None, no_grad=False):
        if cond_dict is None:
            cond_dict = self.get_input(batch)

        print("Generate partially prompted audio with in-context learning")
        # self.model.train()
        # assert self.model.training==True

        target_embeds, target_embeds_attn_mask = (
            cond_dict["crossattn_audiomae_pooled"][0],
            cond_dict["crossattn_audiomae_pooled"][1],
        )

        target_time_steps = target_embeds.size(1)

        (
            input_embeds,
            input_embeds_attn_mask,
            cond_sequence_end_time_idx,
        ) = self.get_input_sequence_and_mask(cond_dict)

        model_input = torch.cat(
            [input_embeds, target_embeds[:, : target_time_steps // 4, :]], dim=1
        )
        model_input_mask = torch.cat(
            [
                input_embeds_attn_mask,
                target_embeds_attn_mask[:, : target_time_steps // 4],
            ],
            dim=1,
        )

        steps = self.mae_token_num

        for _ in range(3 * steps // 4):
            output = self.model(
                inputs_embeds=model_input, attention_mask=model_input_mask
            )["last_hidden_state"]
            # Update the model input
            model_input = torch.cat([model_input, output[:, -1:, :]], dim=1)
            # Update the attention mask
            attention_mask_new_step = torch.ones((model_input_mask.size(0), 1)).to(
                model_input.device
            )
            model_input_mask = torch.cat(
                [model_input_mask, attention_mask_new_step], dim=1
            )

        output = model_input[:, cond_sequence_end_time_idx:]

        return output, cond_dict

    def generate(self, batch, cond_dict=None, no_grad=False):
        if cond_dict is None:
            cond_dict = self.get_input(batch)
        else:
            z = self.get_input(cond_dict)
            cond_dict.update(z)
        # self.model.train()
        # print("!!!!!!!!!!!!!train")

        (
            input_embeds,
            input_embeds_attn_mask,
            cond_sequence_end_time_idx,
        ) = self.get_input_sequence_and_mask(cond_dict)
        model_input = input_embeds
        model_input_mask = input_embeds_attn_mask

        steps = self.mae_token_num

        for _ in range(steps):
            output = self.model(
                inputs_embeds=model_input, attention_mask=model_input_mask
            )["last_hidden_state"]
            # Update the model input
            model_input = torch.cat([model_input, output[:, -1:, :]], dim=1)
            # Update the attention mask
            attention_mask_new_step = torch.ones((model_input_mask.size(0), 1)).to(
                model_input.device
            )
            model_input_mask = torch.cat(
                [model_input_mask, attention_mask_new_step], dim=1
            )

        return model_input[:, cond_sequence_end_time_idx:], cond_dict
    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    

    @torch.no_grad()
    def generate_diffusion(self,src_type,tgt_type,src, 
                           
                           no_grad=False,num_inference_steps=25,
                           eta: float = 0.0,
                           generator=None,
                           image_bind_overwrite=None,
                           guidance_scale=5,
                           score=6.8,
                           negative_score=2.0,
                           do_classifier_free_guidance=True,
                           device='cuda',
                           dtype=torch.float16,
                           no_diffusion=False,
                           force_guidence_t0=False):

                #         timesteps = torch.randint(
                #         0, self.noise_scheduler.config.num_train_timesteps, (target.shape[0],), device=target.device
                #     )
                # noise = torch.randn_like(target)
                # noisy_input = 10.0 * self.noise_scheduler.add_noise(target / 10.0,noise, timesteps)
                # noise_level = get_timestep_embedding(
                #     timesteps=timesteps, embedding_dim=noisy_input.shape[-1], flip_sin_to_cos=True, downscale_freq_shift=0)
                # cond_dict['noisy_input'] = noisy_input
                # cond_dict['noise_level'] = noise_level
        if no_diffusion:
            num_inference_steps = 1
        bs = len(src)
        raw_bs = bs
        if src_type == MODALITY.TEXT:
            src_key = 'text'
        else:
            src_key = 'imagebind'
        if image_bind_overwrite is None:
            image_bind_overwrite = torch.zeros(bs,1,1024).to(device).to(dtype)
        cond_dict = dict(
            src_type = torch.tensor(src_type).view(1,1).to(device).repeat(bs,1),
            tgt_type=torch.tensor(tgt_type).view(1,1).to(device).repeat(bs,1),
            score = get_timestep_embedding(
                torch.tensor([score]).to(device).to(dtype),
                512,flip_sin_to_cos=True,downscale_freq_shift=0
            ).view(1,1,-1).repeat(bs,1,1),
            text=[""],
            imagebind = image_bind_overwrite.to(device).to(dtype),
        )
        if src_key == 'text':
            cond_dict[src_key]= src
        elif src_key == 'imagebind':
            cond_dict[src_key] = src.view(bs,1,-1).to(device).to(dtype)
        if do_classifier_free_guidance:
            cond_dict['src_type'] = cond_dict['src_type'].repeat(2,1)
            cond_dict['tgt_type'] = cond_dict['tgt_type'].repeat(2,1)
            if 'text' in cond_dict:
                cond_dict['text'] = cond_dict['text']+[''] * len(cond_dict['text'])
            if 'imagebind' in cond_dict:
                cond_dict['imagebind'] = torch.cat([cond_dict['imagebind'],cond_dict['imagebind']*0.],dim=0)
            cond_dict['score'] = torch.cat([cond_dict['score'],cond_dict['score']*0.0+negative_score],dim=0)
            bs = bs * 2
        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.noise_scheduler.timesteps
        #print(cond_dict['imagebind'].shape)
        z = self.get_input(cond_dict)
        cond_dict.update(z)
        # self.model.train()
        
        src_type = cond_dict['src_type']
        if no_diffusion:
            noisy_inputs_key = 'noisy_input' # hacl
        else:
            noisy_inputs_key = 'noisy_inputs'
        cond_dict[noisy_inputs_key] = torch.randn(raw_bs,1,self.embed_dim).to(device).to(src_type)
        if do_classifier_free_guidance:
            cond_dict[noisy_inputs_key]  = cond_dict[noisy_inputs_key].repeat(2,1,1)
        #cond_dict['noise_level'] = noise_level
        # print("!!!!!!!!!!!!!train")

        

        steps = self.mae_token_num
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.noise_scheduler.order, 0)
        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                timesteps = torch.ones(
                    (cond_dict[noisy_inputs_key].shape[0],), device=cond_dict[noisy_inputs_key].device
                )* t
                noise_level = get_timestep_embedding(
                     timesteps=timesteps, embedding_dim=cond_dict[noisy_inputs_key].shape[-1], flip_sin_to_cos=True, downscale_freq_shift=0)
                cond_dict['noise_level'] = noise_level
                (
                input_embeds,
                input_embeds_attn_mask,
                cond_sequence_end_time_idx,
                ) = self.get_input_sequence_and_mask(cond_dict)
                model_input = input_embeds
                model_input_mask = input_embeds_attn_mask
                for _ in range(steps):
                    output = self.model(
                        inputs_embeds=model_input, attention_mask=model_input_mask
                    )["last_hidden_state"]
                    # Update the model input
                    model_input = torch.cat([model_input, output[:, -1:, :]], dim=1)
                    # Update the attention mask
                    attention_mask_new_step = torch.ones((model_input_mask.size(0), 1)).to(
                        model_input.device
                    )
                    model_input_mask = torch.cat(
                        [model_input_mask, attention_mask_new_step], dim=1
                    )
                output = model_input[:, cond_sequence_end_time_idx:] #
                #print(output.norm())
                #prev_t = self.noise_scheduler.previous_timestep(t)
                prev_t = self.noise_scheduler.config.num_train_timesteps // self.noise_scheduler.num_inference_steps
                if prev_t >= 0 or force_guidence_t0:
                    output = self.get_eps(t,cond_dict[noisy_inputs_key],output) # to eps transform
                    if do_classifier_free_guidance:
                        noise_pred_text,noise_pred_uncond  = output.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred = output

                    latents = self.noise_scheduler.step(noise_pred, t, cond_dict[noisy_inputs_key][:raw_bs], **extra_step_kwargs, return_dict=False)[0]
                    #print((latents== cond_dict[noisy_inputs_key][:raw_bs]).all())
                    if do_classifier_free_guidance:
                        latents = latents.repeat(2,1,1)
                else:
                    #print("HERE")
                    latents = output[:raw_bs]
                cond_dict[noisy_inputs_key] = latents
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.noise_scheduler.order == 0):
                    progress_bar.update()
        return cond_dict[noisy_inputs_key][:raw_bs], cond_dict

    def forward(self,cond_dict,target=None,use_cache=None):
        if self.diffusion:
            if target is not None:
                timesteps = torch.randint(
                        0, self.noise_scheduler.config.num_train_timesteps, (target.shape[0],), device=target.device
                    )
                noise = torch.randn_like(target)
                noisy_inputs = 10.0 * self.noise_scheduler.add_noise(target / 10.0,noise, timesteps)
                noise_level = get_timestep_embedding(
                    timesteps=timesteps, embedding_dim=noisy_inputs.shape[-1], flip_sin_to_cos=True, downscale_freq_shift=0)
                cond_dict['noisy_inputs'] = noisy_inputs
                cond_dict['noise_level'] = noise_level
            else:
                assert 'noisy_inputs' in cond_dict and 'noise_level' in cond_dict and 'noisy_inputs' in self.sequence_input_key and 'noise_level' in self.sequence_input_key

        z = self.get_input(cond_dict)
        cond_dict.update(z)
        (
            input_embeds,
            input_embeds_attn_mask,
            cond_sequence_end_time_idx,
        ) = self.get_input_sequence_and_mask(cond_dict)
        if target is not None:
            input_embeds = torch.cat([input_embeds,target],dim=1)
            input_embeds_attn_mask = torch.cat([input_embeds_attn_mask,torch.ones(input_embeds_attn_mask.shape[0],target.shape[1]).to(input_embeds_attn_mask)],dim=1)
        output = self.model(
                inputs_embeds=input_embeds, attention_mask=input_embeds_attn_mask,use_cache=use_cache
        )
        if self.training:
            shifted_logits = output['last_hidden_state'][:,-1-target.shape[1]:-1]
            loss = self.loss_fn(shifted_logits,target)
            return loss,output
        return output
        
    def get_input_item(self, batch, k):
        ret = {}

        if "log_mel_spec" in batch:
            ret["fbank"] = (
                batch["log_mel_spec"].unsqueeze(1).to(memory_format=torch.contiguous_format).float()
            )
        if 'stft' in batch:
            ret["stft"] = batch["stft"].to(memory_format=torch.contiguous_format).float()
        # ret["clip_label"] = clip_label.to(memory_format=torch.contiguous_format).float()
        if 'waveform' in batch:
            ret["waveform"] = batch["waveform"].to(memory_format=torch.contiguous_format).float()
        if 'text' in batch:
            ret["text"] = list(batch["text"])
        if 'fname' in batch:
            ret["fname"] = batch["fname"]

        for key in batch.keys():
            if key not in ret.keys():
                ret[key] = batch[key]

        return ret[k]

    def get_input(self, batch):
        cond_dict = {}
        if len(self.cond_stage_model_metadata.keys()) > 0:
            unconditional_cfg = False

            for cond_model_key in self.cond_stage_model_metadata.keys():
                cond_stage_key = self.cond_stage_model_metadata[cond_model_key][
                    "cond_stage_key"
                ]

                # if(not self.training):
                #     if(isinstance(self.cond_stage_models[self.cond_stage_model_metadata[cond_model_key]["model_idx"]], CLAPAudioEmbeddingClassifierFreev2)):
                #         assert cond_stage_key == "text" # CLAP model should use text for evaluation

                # The original data for conditioning
                xc = self.get_input_item(batch, cond_stage_key)
                if type(xc) == torch.Tensor:
                    xc = xc.to(self.device)

                c = self.get_learned_conditioning(
                    xc, key=cond_model_key, unconditional_cfg=unconditional_cfg
                )
                cond_dict[cond_model_key] = c

        return cond_dict

    def instantiate_cond_stage(self, config):
        self.cond_stage_model_metadata = {}

        for i, cond_model_key in enumerate(config.keys()):
            model = instantiate_from_config(config[cond_model_key])
            self.cond_stage_models.append(model)
            self.cond_stage_model_metadata[cond_model_key] = {
                "model_idx": i,
                "cond_stage_key": config[cond_model_key]["cond_stage_key"],
                "conditioning_key": config[cond_model_key]["conditioning_key"],
            }

    def get_learned_conditioning(self, c, key, unconditional_cfg):
        assert key in self.cond_stage_model_metadata.keys()

        # Classifier-free guidance
        if not unconditional_cfg:
            c = self.cond_stage_models[
                self.cond_stage_model_metadata[key]["model_idx"]
            ](c)
        else:
            if isinstance(c, torch.Tensor):
                batchsize = c.size(0)
            elif isinstance(c, list):
                batchsize = len(c)
            else:
                raise NotImplementedError()
            c = self.cond_stage_models[
                self.cond_stage_model_metadata[key]["model_idx"]
            ].get_unconditional_condition(batchsize)

        return c

    def initialize_param_check_toolkit(self):
        self.tracked_steps = 0
        self.param_dict = {}

    def statistic_require_grad_tensor_number(self, module, name=None):
        requires_grad_num = 0
        total_num = 0
        require_grad_tensor = None
        for p in module.parameters():
            if p.requires_grad:
                requires_grad_num += 1
                if require_grad_tensor is None:
                    require_grad_tensor = p
            total_num += 1
        print(
            "Module: [%s] have %s trainable parameters out of %s total parameters (%.2f)"
            % (name, requires_grad_num, total_num, requires_grad_num / total_num)
        )
        return require_grad_tensor
