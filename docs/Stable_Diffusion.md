# Stable Diffusion for AMD GPUs on Windows using DirectML

## Requirements
- Python Installed (https://www.python.org/downloads/)
- Git installed (https://gitforwindows.org/)

## Create a Folder to Store Stable Diffusion Related Files
- Open File Explorer and navigate to your prefered storage location.
- Create a new folder named "Stable Diffusion" and open it.
- In the navigation bar, in file explorer, highlight the folder location and type in cmd and press enter.

## Install 🤗 diffusers
The following steps create a virtual environment (using venv) named sd_env (in the folder you have the cmd window opened to) and then installs diffusers, transformers, onnxruntime, onnx and onnxruntime-directml:
```bash
pip install virtualenv
python -m venv sd_env
sd_env\scripts\activate
pip install diffusers
pip install transformers
pip install onnxruntime
pip install onnx
pip install torch
pip install onnxruntime-directml --force-reinstall
```
To exit the virtual environment, close the command prompt. To start the virtual environment go to the scripts folder in sd_env and open a command prompt. Type activate and the virtual environment will activate.

## Download the Stable Diffusion ONNX model
![red-stop-icon](https://user-images.githubusercontent.com/640619/197152731-c46f3f88-5fab-4c76-bfc3-e85d33eb7593.png)

You will need to go to: https://huggingface.co/runwayml/stable-diffusion-v1-5 and https://huggingface.co/runwayml/stable-diffusion-inpainting. Review and accept the usage/download agreements before completing the following steps.

- stable-diffusion-v1-5 uses 5.10 GB
- stable-diffusion-inpainting uses 5.10 GB

If your model folders are larger, open stable_diffusion_onnx and stable_diffusion_onnx_inpainting and delete the .git folders

```bash
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 --branch onnx --single-branch stable_diffusion_onnx
git clone https://huggingface.co/runwayml/stable-diffusion-inpainting --branch onnx --single-branch stable_diffusion_onnx_inpainting
```
Enter in your HuggingFace credentials and the download will start.
Once complete, you are ready to start using Stable Diffusion

## Stable Diffusion Txt 2 Img on AMD GPUs

Here is an example python code for the Onnx Stable Diffusion Pipeline using huggingface diffusers.

```python
from diffusers import OnnxStableDiffusionPipeline
height=512
width=512
num_inference_steps=50
guidance_scale=7.5
eta=0.0
prompt = "a photo of an astronaut riding a horse on mars"
negative_prompt="bad hands, blurry"
pipe = OnnxStableDiffusionPipeline.from_pretrained("./stable_diffusion_onnx", provider="DmlExecutionProvider")
image = pipe(prompt, height, width, num_inference_steps, guidance_scale, negative_prompt, eta).images[0] 
image.save("astronaut_rides_horse.png")
```
![image](https://user-images.githubusercontent.com/640619/197129879-13aeca7a-a3b6-40b8-b5ca-54f8bfc29408.png)

## Stable Diffusion Img 2 Img on AMD GPUs

Here is an example python code for Onnx Stable Diffusion Img2Img Pipeline using huggingface diffusers.

```python
import time
import torch
from PIL import Image
from diffusers import OnnxStableDiffusionImg2ImgPipeline

init_image = Image.open("test.png")
prompt = "A fantasy landscape, trending on artstation"

pipe = OnnxStableDiffusionImg2ImgPipeline.from_pretrained("./stable_diffusion_onnx", provider="DmlExecutionProvider", revision="fp16", torch_dtype=torch.float16)
image = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5).images[0] 
image.save("test-output.png")
```

## Stable Diffusion Inpainting on AMD GPUs

Here is an example python code for the Onnx Stable Diffusion Inpaint Pipeline using huggingface diffusers.

```python
import torch
from PIL import Image
from diffusers import OnnxStableDiffusionInpaintPipeline

pipe = OnnxStableDiffusionInpaintPipeline.from_pretrained("./stable_diffusion_onnx_inpainting", provider="DmlExecutionProvider", revision="fp16", torch_dtype=torch.float16)
pipe.safety_checker = lambda images, **kwargs: (images, False)

init_image = Image.open("test.png")
init_image = init_image.resize((512, 512))
mask_image = Image.open("mask.png")
mask_image = mask_image.resize((512, 512))
prompt = "Face of a yellow cat, high resolution"

image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.75, guidance_scale=7.5).images[0] 
image.save("test-output.png")
```

Inpaint Images need to be width 512 height 512

You can make an image mask using https://www.photopea.com/

## Example Txt2Img Script With More Features
- User is prompted in console for Image Parameters
- Date/Time, Image Parameters & Completion Time is logged in a Txt File "prompts.txt"
- Image is saved, named date-time.png (date-time = time image generation was started)
- User is asked for another prompt or q to quit.

```python
import os
import sys
import time
import torch
import numpy as np
from diffusers import OnnxStableDiffusionPipeline

height=512
width=512
num_inference_steps=50
guidance_scale=7.5
eta=0.0
prompt=""
variations=""
negative_prompt=""
model=""
pipe=""

os.system('cls')
print('Stable Diffusion Onnx DirectML\nText to Img\n')
while model == "":
    model = input('Avalible Models\n1 (sd_1.5)\nPlease Choose a Model#: (or q to quit): ')
    if model == "q":
        os.system('cls')
        sys.exit("Quit Called, Script Ended")
    elif model == "1":
        model = "./stable_diffusion_onnx"
    else:
        model = ""

os.system('cls')
pipe = OnnxStableDiffusionPipeline.from_pretrained(model, provider="DmlExecutionProvider", revision="fp16", torch_dtype=torch.float16)
#pipe.safety_checker = lambda images, **kwargs: (images, False)

def txt_to_img(prompt, negative_prompt, num_inference_steps, width, height, seed):
    gen_time = time.strftime("%m%d%Y-%H%M%S")
    generator = torch.Generator()
    if seed == "":
        seed = generator.seed()
    else:
        seed = int(seed)
    generator = generator.manual_seed(seed)
    latents = torch.randn(
        (1, 4, height // 8, width // 8),
        generator = generator
    )
    start_time = time.time()
    image = pipe(prompt, height, width, num_inference_steps, guidance_scale, negative_prompt, eta, latents = latents, execution_provider="DmlExecutionProvider").images[0] 
    image.save("./" + gen_time + ".png")
    log_info = "\n" + gen_time + " - Seed: " + str(seed) + " - Gen Time: "+ str(time.time() - start_time) + "s"
    with open('./prompts.txt', 'a+', encoding="utf-16") as f:
        f.write(log_info)
    image = None

while prompt != "q":
    os.system('cls')
    print('Stable Diffusion Onnx DirectML (' + model + ')\nText to Img\n')
    while prompt == "":
        prompt = input('Please Enter Prompt (or q to quit): ')
    if prompt != "q":
        negative_prompt = input('Please Enter Negative Prompt (Optional): ')
        while variations == "":
            variations = input('How Many Images? (Optional): ')
            if variations.isnumeric() == False:
                variations = ""
            if variations == 0 or variations == "":
                variations = "1"
        num_inference_steps = input('Please Enter # of Inference Steps (Optional): ')
        if num_inference_steps.isnumeric() == False:
            num_inference_steps = 50
        width = input('Please Enter Width 512 728 768 1024 (Optional): ')
        if width.isnumeric() == False:
            width = 512
        height = input('Please Enter Height 512 728 768 1024 (Optional): ')
        if height.isnumeric() == False:
            height = 512
        seed = input('Please Enter Seed (Optional): ')
        if seed.isnumeric() == False:
            seed = ""
        gen_time = time.strftime("%m%d%Y-%H%M%S")
        log_info = "\n" + gen_time + " - Model: " + model
        log_info += "\n" + gen_time + " - Prompt: " + prompt
        log_info += "\n" + gen_time + " - Neg_Prompt: " + negative_prompt
        log_info += "\n" + gen_time + " - Inference Steps: " + str(num_inference_steps) + " Guidance Scale: " + str(guidance_scale) + " Width: " + str(width) + " Height: " + str(height)
        with open('./prompts.txt', 'a+', encoding="utf-16") as f:
            f.write(log_info)
        os.system('cls')
        for i in range(int(variations)):
            print(str(i) + "/" + str(variations))
            txt_to_img(prompt, negative_prompt, int(num_inference_steps), int(width), int(height), seed)
        prompt = ""
        variations = ""
pipe = None
os.system('cls')
sys.exit("Quit Called, Script Ended")
```

### Output
prompts.txt
```
10232022-233730 - Model: ./stable_diffusion_onnx
10232022-233730 - Prompt: cat
10232022-233730 - Neg_Prompt: dog
10232022-233730 - Inference Steps: 50 Guidance Scale: 7.5 Width: 512 Height: 512
10232022-233730 - Seed: 22220167420300 - Gen Time: 250.15623688697815s
```

![image](https://user-images.githubusercontent.com/640619/197464340-01faac49-6d41-46b8-bd97-2d46af2d0e31.png)