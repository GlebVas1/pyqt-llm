# https://stackoverflow.com/questions/77630013/how-to-run-any-quantized-gguf-model-on-cpu-for-local-inference

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

def downloadModel(modelName : str = "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"):
    fileName = "./models/" + modelName.split('/')[1]
    filePath = hf_hub_download(modelName, filename=fileName)



    