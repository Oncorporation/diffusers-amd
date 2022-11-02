import os
from diffusers import OnnxStableDiffusionImg2ImgPipeline
from PIL import Image
from pathlib import Path
import time
import torch
import numpy as np
import gc
import sys

def get_latents_from_seed(seed: int, width: int, height:int) -> np.ndarray:
    # 1 is batch size
    latents_shape = (1, 4, height // 8, width // 8)
    # Gotta use numpy instead of torch, because torch's randn() doesn't support DML
    rng = np.random.default_rng(seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents

height = 592
width = 592
size = 0

print('Stable Diffusion Onnx DirectML\nImg to Img\n')
localuserPath = os.path.expanduser('~')
baseImagePath = Path(localuserPath + r"\Saved Games\MechWarrior Online\UI\MechIcons")
baseImageName = r"271725400949035.png"
filename = ""
while filename == "":
    filename = input('Please Enter FileName (or q to quit): ')
if filename != "q":
    while size == 0:
        size = input('image size? (' + str(height) + '): ')
        if size.isnumeric() == False:
            size = height
        width = int(size)
        height = int(size)
        baseImageName = filename
else:
    sys.exit("You have quit entering filenames")

baseImage = Image.open(str(baseImagePath) + "\\" + baseImageName).convert("RGB")
baseImage = baseImage.resize((height,width))

#prompt = "a photorealistic portrait, centered, upper body and hips, facing slight left, firework transparent glass illuminated statue of (gigantic battletech mechwarrior in matte black with three (glossy red hyperdetailed circular, chinese dragon insignia in style of vector art), beautiful mecha), realistic, unreal engine, 4 k, symmetrical, hyperdetailed, micro details, inside aircraft hangar"
#prompt = "a photorealistic, looking to his left:5, portrait of a firework luminous glowing (gigantic battletech mechwarrior with three (glossy red hyperdetailed circular, serpentine dragon insignia in style of vector art), symmetrical) in a futuristic hangar, 8 k, micro details, by unreal engine, centered"
#prompt = "firework style, a photorealistic camera shot, portrait looking to his left of a glowing illuminated (gigantic battletech mechwarrior mecha), symmetrical) in a futuristic hanger, bright studio setting, studio lighting, crisp quality and light reflections, unreal engine 5 quality render, 8k, micro-details, volumetric"
prompt = "retrowave epic art, clipped right, firework style, centered, (a photorealistic camera shot portrait in profile left) of a glowing illuminated (gigantic battletech mechwarrior mecha, guns), symmetrical) in a futuristic hangar, bright studio setting, studio lighting, crisp quality and light reflections, unreal engine 5 quality render, 8k, micro-details, volumetric"
#1prompt = "retrowave epic art, clipped right, centered, (a photorealistic camera shot close-up portrait looking to his left):10 of a illuminated (gigantic battletech mechwarrior mecha) with minigun and lasers), symmetrical) in red glowing hangar, bright studio setting, studio lighting, crisp quality and light reflections, unreal engine 5 quality render, 8k, volumetric"
#3prompt = "clipped right, centered, intricate oil painting of a (giant prestine white mechsuit mecha mechwarrior) with (minigun and (rocket launchers) and laser) by simon stalenhaq, by ian mcque, in (red glowing futuristic hangar), bright studio setting, studio lighting, crisp quality and light reflections, 8k, volumetric, hardsurface modeling"
#4prompt = "clipped right, centered, (a photorealistic camera shot close-up portrait looking to his left):10 of a ((gigantic battletech mechwarrior war mech) with minigun) in red glowing hangar, studio lighting, crisp quality and light reflections, 8k, hardsurface modeling, hyperdetailed, by iam mcque, by simon stalenhag"
#5prompt = "retrowave epic art, clipped right, centered, (a photorealistic camera shot portrait in profile left):10 of a illuminated (gigantic battletech mechwarrior mecha) with minigun and lasers), symmetrical) in red glowing hangar, bright studio setting, studio lighting, crisp quality and light reflections, unreal engine 5 quality render, 8k, volumetric"

denoiseStrength = 0.44
steps = 68
scale = 14
negprompt = ""
eta = 0.01
num_images_per_prompt = 1
seed = int(baseImageName.split('.')[0][0:10])
if seed > 4294967295:
    seed / 10
latents = get_latents_from_seed(seed, height, width)
onnxmodel = Path("G:\Projects\Stable-Diffusion-webui-amd\models\onnx")

pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(onnxmodel, provider="DmlExecutionProvider", revision="fp16", torch_dtype=torch.float16)
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

image = pipe(prompt, height=height, width=width, init_image=baseImage, eta=eta, strength=denoiseStrength, num_inference_steps=steps, num_images_per_prompt=num_images_per_prompt, guidance_scale=scale, negative_prompt=negprompt, latents=latents).images[0]
image.save("./output/"+ str(height) + "-" + baseImageName)
resized = image.resize((384, 384), Image.Resampling.LANCZOS)
resized.save("./output/" + baseImageName)

image = None
resized = None
pipe = None
gc.collect()
print('files saved in output subfolder Img\n')