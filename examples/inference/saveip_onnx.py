from pathlib import Path
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import torch
from transformers import CLIPTextModel
from optimum.onnxruntime import ORTModelForSequenceClassification


text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", return_dict=False)

torch.manual_seed(8008)
lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

v14 = "CompVis/stable-diffusion-v1-4"
v15 = "runwayml/stable-diffusion-v1-5"
v15pd = "g://projects//stable-diffusion-webui-amd//models//Stable-diffusion//v1-5-pruned.ckpt"
v15inpd = "g://projects//stable-diffusion-webui-amd//models//Stable-diffusion//sd-v1-5-inpainting.ckpt"
v15inp = "runwayml/stable-diffusion-inpainting"
 
pipe = StableDiffusionPipeline.from_pretrained(v15inp, scheduler=lms, use_auth_token=True)

def convert_to_onnx(unet, post_quant_conv, decoder, text_encoder, height=512, width=512):
    """Convert given input models to onnx files.
        unet: UNet2DConditionModel
        post_quant_conv: AutoencoderKL.post_quant_conv
        decoder: AutoencoderKL.decoder
        text_encoder: CLIPTextModel
        feature_extractor: TODO
        safetychecker: TODO
        height: Int
        width: Int
        Note: 
            - opset_version required is 15 for CLIPTextModel
    """
    p = Path("onnx/")
    p.mkdir(parents=True, exist_ok=True)

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
    h, w = height // 8, width // 8
    # unet onnx export
    check_inputs = [(torch.rand(2, 4, h, w), torch.tensor([980], dtype=torch.long), torch.rand(2, 77, 768)), 
                    (torch.rand(2, 4, h, w), torch.tensor([910], dtype=torch.long), torch.rand(2, 12, 768)), # batch change, text embed with no trunc
                    ]
    traced_model = torch.jit.trace(unet, check_inputs[0], check_inputs=[check_inputs[1]], strict=True)
    torch.onnx.export(traced_model, check_inputs[0], "onnx/unet.onnx", 
        input_names=["latent_model_input", "t", "encoder_hidden_states"], 
        dynamic_axes={ "latent_model_input": [0], "t": [0], "encoder_hidden_states": [0, 1]}, opset_version=16)

    # post_quant_conv onnx export
    check_inputs = [(torch.rand(1, 4, h, w),), (torch.rand(2, 4, h, w),)]
    traced_model = torch.jit.trace(post_quant_conv, check_inputs[0], check_inputs=[check_inputs[1]])
    torch.onnx.export(traced_model, check_inputs[0], "onnx/post_quant_conv.onnx", 
        input_names=["latents"], dynamic_axes={"latents": [0]}, opset_version=16)

    # decoder onnx export
    check_inputs = [(torch.rand(1, 4, h, w),), (torch.rand(2, 4, h, w),)]
    traced_model = torch.jit.trace(decoder, check_inputs[0], check_inputs=[check_inputs[1]])
    torch.onnx.export(traced_model, check_inputs[0], "onnx/decoder.onnx", 
        input_names=["latents"], dynamic_axes={"latents": [0]}, opset_version=16)

    # encoder onnx export
    check_inputs = [(torch.randint(1, 24000, (1, 77)),), (torch.randint(1, 24000, (2, 77)),)]
    traced_model = torch.jit.trace(text_encoder, check_inputs[0], check_inputs=[check_inputs[1]], strict=False)
    torch.onnx.export(traced_model, check_inputs[0], "onnx/encoder.onnx", input_names=["text_input"], 
        dynamic_axes={"text_input": [0, 1]}, opset_version=16)
    

# Change height and width to create ONNX model file for that size
convert_to_onnx(pipe.unet, pipe.vae.post_quant_conv, pipe.vae.decoder, text_encoder, height=512, width=512)

# For example, create an onnx model with height=512 and width=768
# convert_to_onnx(pipe.unet, pipe.vae.post_quant_conv, pipe.vae.decoder, text_encoder, height=512, width=768)
