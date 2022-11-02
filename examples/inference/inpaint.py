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