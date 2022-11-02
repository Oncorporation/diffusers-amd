g:
cd G:\Projects\Stable-Diffusion-webui-amd
rem pip install virtualenv
py -m venv venv
call venv\scripts\activate
rem pip install diffusers
rem pip install transformers
rem pip install onnxruntime
rem pip install onnx
rem pip install torch
rem pip install onnxruntime-directml --force-reinstall




py venv/scripts/pip.exe install diffusers==0.6.0 
py venv/scripts/pip.exe install transformers[onnx]
rem py venv/scripts/pip.exe install onnxruntime
py venv/scripts/pip.exe install optimum[onnxruntime]
py venv/scripts/pip.exe install onnx
py venv/scripts/pip.exe install torch
py venv/scripts/pip.exe install onnxruntime-directml --force-reinstall
rem py ./venv/scripts/pip.exe install g://projects//stable-diffusion-webui-amd//ort_nightly_directml-1.13.0.dev20221021004-cp310-cp310-win_amd64.whl --force-reinstall
cd G:\Projects\Stable-Diffusion-webui-amd\diffusers\examples\inference