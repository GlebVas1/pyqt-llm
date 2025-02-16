# https://stackoverflow.com/questions/77630013/how-to-run-any-quantized-gguf-model-on-cpu-for-local-inference

from huggingface_hub import hf_hub_download, snapshot_download
from llama_cpp import Llama

def downloadModel(modelName : str = "intfloat/multilingual-e5-large-instruct"):
    filePath = snapshot_download(modelName, local_dir="./models/downloaded/" + modelName.split('/')[1], local_dir_use_symlinks=False, revision="main")


downloadModel()
