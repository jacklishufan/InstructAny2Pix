from .model import InstructAny2PixPrior
prior_config ={
        "always_output_audiomae_gt": False,
        "learnable": True,
        "device": "cuda",
        "use_gt_mae_output": True,
        "use_gt_mae_prob": 0.0,
        "base_learning_rate": 0.0002,
        "sequence_gen_length": 1,
        "diffusion": True,
        "use_warmup": True,
        "sequence_input_key": [
            "src_type",
            "imagebind",
            # "crossattn_flan_t5",
            "crossattn_clip",
            "score",
            "noisy_inputs",
            "noise_level"
            "tgt_type",
        ],
        "sequence_input_embed_dim": [0,1024,1024,512,0,0,0],
        "pretrained_name":'gpt2-medium',
        "batchsize": 16,
        "embed_dim":1024,
        "output_dim":1024,
        "cond_stage_config": {
            # "crossattn_flan_t5": {
            #     "cond_stage_key": "text",
            #     "conditioning_key": "crossattn",
            #     "target": "audioldm2.latent_diffusion.modules.encoders.modules.FlanT5HiddenState",
            # },
            "crossattn_clip": {
                "cond_stage_key": "text",
                "conditioning_key": "crossattn",
                "target": "instructany2pix.prior.model.CLIPTextModelHiddenState",
            },

            
        },
}