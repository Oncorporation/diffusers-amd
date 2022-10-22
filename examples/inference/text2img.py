from diffusers import StableDiffusionOnnxPipeline
import numpy as np

def get_latents_from_seed(seed: int, width: int, height:int) -> np.ndarray:
    # 1 is batch size
    latents_shape = (1, 4, height // 8, width // 8)
    # Gotta use numpy instead of torch, because torch's randn() doesn't support DML
    rng = np.random.default_rng(seed)
    image_latents = rng.standard_normal(latents_shape).astype(np.float32)
    return image_latents

pipe = StableDiffusionOnnxPipeline.from_pretrained("./stable_diffusion_onnx", provider="DmlExecutionProvider")
"""
prompt: Union[str, List[str]],
height: Optional[int] = 512,
width: Optional[int] = 512,
num_inference_steps: Optional[int] = 50,
guidance_scale: Optional[float] = 7.5, # This is also sometimes called the CFG value
eta: Optional[float] = 0.0,
latents: Optional[np.ndarray] = None,
output_type: Optional[str] = "pil",
"""
pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
seed = 50033
prompt = "mos eisley cantina, a tiny transparent glass illuminates statue [a full body shot photo of a translucent [Princess Leia], movie still, [halluzinogenic, translucent!, Opalescent crystal, transparent!, glass skin, blue illumination, blue glow]:0.7] 'on top of table'"

image = pipe(prompt, num_inference_steps=25, guidance_scale=13, latents=latents).images[0]
image.save("PL25.png")