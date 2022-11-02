import os
from diffusers import OnnxStableDiffusionImg2ImgPipeline
from PIL import Image
from pathlib import Path
import time
import torch
import numpy as np
import gc

def get_latents_from_seed(seed: int, width: int, height:int) -> np.ndarray:
    # 1 is batch size
    latents_shape = (1, 4, height // 8, width // 8)
    # Gotta use numpy instead of torch, because torch's randn() doesn't support DML
    rng = np.random.default_rng(seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents

def i2i_update(baseImageName: str, baseImagePath: str, outputImagePath: str, width: int, height:int, pipe):
    baseImage = Image.open(str(baseImagePath) + "\\" + baseImageName).convert("RGB")
    baseImage = baseImage.resize((height,width))

    #prompt = "a photorealistic portrait, centered, upper body and hips, facing slight left, firework illuminated (gigantic battletech mechwarrior with three (glossy red hyperdetailed circular, chinese dragon insignia in style of vector art), symmetrical), realistic, unreal engine, 4 k, hyperdetailed, micro details, inside aircraft hangar"
    #prompt = "a photorealistic, facing slight left, portrait of a firework luminous glowing (gigantic battletech mechwarrior with three (glossy red hyperdetailed circular, serpentine dragon insignia in style of vector art), symmetrical) in a futuristic hangar, 8 k, micro details, by unreal engine, centered"
    #prompt = "firework style, a photorealistic camera shot, portrait facing slight left of a glowing illuminated (gigantic battletech mechwarrior mecha), symmetrical) in a futuristic hangar, bright studio setting, studio lighting, crisp quality and light reflections, unreal engine 5 quality render, 8k, micro-details, volumetric, clipped on left"
    prompt = "retrowave epic art, clipped right, firework style, centered, (a photorealistic camera shot portrait in profile left) of a glowing illuminated (gigantic battletech mechwarrior mecha, guns), symmetrical) in a futuristic hangar, bright studio setting, studio lighting, crisp quality and light reflections, unreal engine 5 quality render, 8k, micro-details, volumetric"
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

    image = pipe(prompt, height=height, width=width, init_image=baseImage, eta=eta, strength=denoiseStrength, num_inference_steps=steps, num_images_per_prompt=num_images_per_prompt, guidance_scale=scale, negative_prompt=negprompt, latents=latents).images[0]
    image.save(str(outputImagePath) + "\\" + str(height) + "-" + baseImageName)
    resized = image.resize((384, 384), Image.Resampling.LANCZOS)
    resized.save(str(outputImagePath) + "\\" + baseImageName)
    image = None
    resized = None
    pipe = None
    gc.collect()

onnxmodel = Path("G:\Projects\Stable-Diffusion-webui-amd\models\onnx")

pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained(onnxmodel, provider="DmlExecutionProvider", revision="fp16", torch_dtype=torch.float16)
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

height = 592
width = 592
batch = 1

outputImagePath = Path(r"J:\AI\MechIcons")
localuserPath = os.path.expanduser('~')
baseImagePath = Path(localuserPath + r"\Saved Games\MechWarrior Online\UI\MechIcons")

paths = sorted(Path(baseImagePath).iterdir(), key=os.path.getmtime, reverse=False)
for path in paths[:25]:
    print('\nProcessing Image#' + str(batch) + ' ' + path.name + '\n')
    i2i_update(path.name, baseImagePath, outputImagePath, height, width, pipe)
    gc.collect()
    batch += 1
    time.sleep(60)