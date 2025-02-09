from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss

class Parameters():
    splitChunkSize = 2000
    splitChunkOverlap = 20


class mainModel(Parameters):

    mainModelModels = {}

    def mainModelInitializeModel(self, name : str = "1") -> None:
        pass

    def loadModelfromFile(self, path : str = "./models/Llama-3.1-Tulu-3-8B-Q8_0.gguf") -> None:
        self.llm = Llama(
            model_path=path,
            n_ctx=2000,  # Context length to use
            n_threads=54,            # Number of CPU threads to use
            n_gpu_layers=0        # Number of model layers to offload to GPU
        )

        embed_model_path = "./models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"
        self.llmEmbeding = LlamaCppEmbeddings(
            model_path=embed_model_path,
            n_ctx=2000,  # Context length to use
            n_threads=54,            # Number of CPU threads to use
            n_gpu_layers=0        # Number of model layers to offload to GPU
        )

        self.generationKwargs = {
            "max_tokens":1000,
            "stop":["</s>"],
            "echo":False, # Echo the prompt in the output
            "top_k":1 # This is essentiallys greedy decoding, since the model will always return the highest-probability token. Set this value > 1 for sampling decoding
        }
    
    def mainModelSplitAndEmbedText(self, text : str) -> None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.splitChunkSize, chunk_overlap=self.splitChunkOverlap)
        self.splittedText = splitter.split_text(text)
        print(self.splittedText)
        print(len(self.splittedText))
        textsEmbeds = []
        for text in self.splittedText:
            # print(text)
            # print(self.llmEmbeding.embed_query(text))
            resultArray = self.llmEmbeding.embed_query(text)
            # print(resultArray)
            textsEmbeds.append(resultArray)
        
        #print(textsEmbeds)
        textsEmbeds = np.array(textsEmbeds)
        print(textsEmbeds.shape)
        print(textsEmbeds)

        d = textsEmbeds.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(textsEmbeds)

        question = "Что такое стратегии объектно-реляционного отображения: систематизация и анализ на основе паттернов?"

        question_embedding = np.array([self.llmEmbeding.embed_query(question)])

        D, I = index.search(question_embedding, k=2)

        retrieved_chunk = [self.splittedText[i] for i in I.tolist()[0]]
        
        request = f'''Ответь на вопрос: {question} 
                      Используя только заданный контекст {retrieved_chunk}'''
        
        result = self.llm(prompt=request, **self.generationKwargs)


        print(retrieved_chunk)
        print(result)


    def mainModelComputeRequest(self, request : str) -> str:

        request = f'''Ответь на вопрос: {request} '''
        
        result = self.llm(prompt=request, **self.generationKwargs)
        return result["choices"][0]["text"]

    def mainModelSetContextFile(self, path : str) -> None:
        pass

    def mainModelDownload(self, name : str) -> None:
        pass
