
""" stable diffusion 2.1 """

import sys
sys.path.insert(0, '/media/tx-deepocean/data_1/yzhen/fun/myenvs/torch')
sys.path.append('/media/tx-deepocean/data_1/yzhen/fun/myenvs/pkgs')

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

def generate_img(prompt, art_i, n_imgs, fn, out_path, height=768, width=1024):
    """ stable difussion """
    model_id = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda:3")
    
    ## 单张图片
    for i in range(n_imgs):
        image = pipe(prompt, height=768, width=1024).images[0]
        image.save(f"./imgs/{art_i}{i}.png")

artists = ['Vincent van Gogh', 'Claude Monet', 'Edgar Degas', 'Pierre-Auguste Renoir',
           'Paul Cézanne', 'Georges Seurat', 'Camille Pissarro', 'Félix Vallotton',
           'Henri Matisse', 'Paul Gauguin', 'Henri Toulouse-Lautrec', 'Edmond de Goncourt', 
           'James Ensor', 'Maurice de Vlaminck', 'Georges Braque', 'James McNeill Whistler', 
           'Auguste Rodin']

# prompt = 'On a hot afternoon, there is a forest next to a big mountain. Many beehives under the forest. A couple who keep bees are checking the beehives under the tree. There is also a big tent where they live and two dogs next to the forest.'
prompt = 'A hot summer afternoon, with a breeze, surrounded by farmland and lakes, a young man sleep quietly on a pile of green grass under trees.'

for i in [6]:
    prompt_ = prompt + f" By {artists[i]}."
    generate_img(prompt_, i, 25, None, None)