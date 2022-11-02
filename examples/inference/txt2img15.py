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
num_inference_steps = 30
guidance_scale = 7.5
eta = 0.0
prompt=""
negprompt=""
variations=""
onnxmodel = Path("G:\Projects\Stable-Diffusion-webui-amd\models\onnx")
pipe = OnnxStableDiffusionPipeline.from_pretrained(onnxmodel, provider="DmlExecutionProvider", torch_dtype=torch.float16)
#pipe = StableDiffusionOnnxPipeline.from_pretrained(onnxmodel, provider="DmlExecutionProvider", torch_dtype=torch.float16)
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

def txt_to_img(prompt):
    generator = torch.Generator()
    seed = generator.seed()
    generator = generator.manual_seed(seed)
    latents = torch.randn(
        (1, 4, height // 8, width // 8),
        generator = generator
    )
    gen_time = time.strftime("%m%d%Y-%H%M%S")
    start_time = time.time()
    image = pipe(prompt, height, width, num_inference_steps, guidance_scale, negprompt, eta, latents = latents, execution_provider="DmlExecutionProvider").images[0] 
    image.save("./" + gen_time + ".png")
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
        while variations == "":
            variations = input('How many image variations?: ')
            if variations.isnumeric() == False:
                variations = ""
        for i in range(int(variations)):
            txt_to_img(prompt)
        prompt = ""
        variations = ""
pipe = None
os.system('cls')
sys.exit("Quit Called, Script Ended")