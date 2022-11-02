import os
import sys
import time
import torch
import numpy as np
#from diffusers import StableDiffusionOnnxPipeline
from diffusers import OnnxStableDiffusionPipeline
from pathlib import Path

height = 512
width = 512
num_inference_steps = 32
guidance_scale = 7.5
eta = 0.0
prompt=""
negprompt=""
variations=""
size = 512
onnxmodel = Path("G:\Projects\Stable-Diffusion-webui-amd\models\onnx")
pipe = OnnxStableDiffusionPipeline.from_pretrained(onnxmodel, provider="DmlExecutionProvider", torch_dtype=torch.float16)
#pipe = StableDiffusionOnnxPipeline.from_pretrained(onnxmodel, provider="DmlExecutionProvider", torch_dtype=torch.float16)
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

def txt_to_img(prompt, size):
    generator = torch.Generator()
    seed = generator.seed()
    generator = generator.manual_seed(seed)
    latents = torch.randn(
        (1, 4, size // 8, size // 8),
        generator = generator
    )
    gen_time = time.strftime("%m%d%Y-%H%M%S")
    start_time = time.time()
    image = pipe(prompt, size, size, num_inference_steps, guidance_scale, negprompt, eta, latents = latents, execution_provider="DmlExecutionProvider").images[0] 
    image.save("./output/" + gen_time + ".png")
    log_info = "\n" + gen_time + " - Prompt: " + prompt + " - Seed: " + str(seed) + " (" + str(time.time() - start_time) + "s)"
    with open('./prompts.txt', 'a+', encoding="utf-8") as f:
        f.write(log_info)
    image = None

os.system('cls')
print('Stable Diffusion Onnx DirectML\nText to Img\n')
while prompt != "q":
    while prompt == "":
        prompt = input('Please Enter Prompt (or q to quit): ')
    if prompt != "q":
        while size == 512:
            size = input('image size? (default 512): ')
            if size.isnumeric() == False:
                size = 512
        while variations == "":
            variations = input('How many image variations?: ')
            if variations.isnumeric() == False:
                variations = ""
        for i in range(int(variations)):
            txt_to_img(prompt, int(size))
        prompt = ""
        variations = ""
        size = 512
pipe = None
os.system('cls')
sys.exit("Quit Called, Script Ended")