import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDMPSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(prompt: str, uncond_prompt: str, input_image=None, strength=0.8, do_cfg=True, cfg_scale=7.5, sampler_name="ddpm", n_inference_steps=50, models={}, seed=None,
            device=None, idle_device=None, tokenizer=None):
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
        generator = torch.Generator(device=device)
        if seed is None:
            generate.seed()
        else:
            generator.manual_seed(seed)
        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            #convert prompt into tokrnd using the tokernizer
            cond_tokens=tokenizer.batch_encode_plus([prompt], padding ='max length', max_length=77).input_ids
            #(batch_size, seq_len)
            cond_tokens= torch.tensor(cond_tokens, dtype=torch.long, device=None)
            #(batch_size, seq_len ) -> (batch_size, seq_len ,dim)
            cond_context= clip(cond_tokens)

            uncond_tokens= tokenizer.batch_encode_plus([uncond_prompt], padding = 'max_length', max_length=77).input_ids
            uncond_tokens= torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            #(batch_size, seq_len) -> (batch_size , seq_len ,dim)
            uncond_context= clip(uncond_tokens)

            #(2, seq_len, dim) = (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else:
            #convert it into list of tokens
            tokens=tokenizer.batch_encode_plus([prompt], padding ='max length', max_length=77).input_ids
            tokens= torch.tensor(tokens, dtype= torch.long, device=device)
            #(1,77,768)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name =='ddpm':
            sampler = DDMPSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError("unknown sampler {sampler_name}")
        
        latents_shape= (1,4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image: 
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor= input_image.resize((WIDTH, HEIGHT))
            input_image_tensor= np.array(input_image_tensor)
            #(height , wodth, channel)
            input_image_tensor= torch.tensor(input_image_tensor, dtype=torch.float32)
            