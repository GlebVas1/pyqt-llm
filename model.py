from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss

class Parameters():
    splitChunkSize = 800
    splitChunkOverlap = 100
    embeddingGuffModel = False


class mainModel(Parameters):

    usedAnswerModel = None
    usedEmbeddingModel = None

    def InitializeModel(self, name : str = "1") -> None:
        pass

    def LoadEmbeddingModelFromFile(self, path : str = "./models/multilingual-e5-large-instruct_q8_0.gguf") -> None:
        self.llmEmbeding = LlamaCppEmbeddings(
            model_path=path,
            n_ctx=2000,  # Context length to use
            n_threads=54,            # Number of CPU threads to use
            n_gpu_layers=0        # Number of model layers to offload to GPU
        )
        self.usedEmbeddingModel = path.split("/")[-1].split(".")[0]




    def LoadAndSaveGenerationKwargs(self) -> None:
        self.generationKwargs = {
            "max_tokens":1000,
            "stop":["</s>"],
            "echo":False, # Echo the prompt in the output
            "top_k":1 # This is essentiallys greedy decoding, since the model will always return the highest-probability token. Set this value > 1 for sampling decoding
        }

    def LoadAnswerModelFromFile(self, path : str = "./models/Llama-3.1-Tulu-3-8B-Q8_0.gguf") -> None:
        self.llm = Llama(
            model_path=path,
            n_ctx=2000,  # Context length to use
            n_threads=54,            # Number of CPU threads to use
            n_gpu_layers=0        # Number of model layers to offload to GPU
        )
       
    def SplitText(self, text, chunk_size : int, chunk_overlap : int) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)

    def EmbedTexts(self, splittedText : list[str]) -> None:
        textsEmbeds = []
        for text in splittedText:
            # Because of embed queue cometimes incorrectly parse model results in it
            resultArray = self.llmEmbeding.client.embed(text)
            textsEmbeds.append(resultArray)
        
        textsEmbeds = np.array(textsEmbeds)

        d = textsEmbeds.shape[1]
        self.RAGIndex = faiss.IndexFlatL2(d)
        self.RAGIndex.add(textsEmbeds)

    def SaveEmbededVectorStorage(self, path : str = "./vector_db/") -> None:
        path += self.usedEmbeddingModel
        self.RAGIndex.save


    def EmbedQuestion(question : str):

        question = "Who is Valakas"
        question = "Кто такой Валакас?"
        question = "Типы данных, не относящиеся к ОО расширениям, но позволяющие хранить неатомарные значения?"
        question_embedding = np.array([self.llmEmbeding.client.embed(question)])

        D, I = index.search(question_embedding, k=2)

        retrieved_chunk = [self.splittedText[i] for i in I.tolist()[0]]
        
        # request = f'''You are giving the following contest
        #             ---------------------
        #             {retrieved_chunk}
        #             ---------------------
        #             Answer the question using only the above provided information
        #             Question: {question}
        #             Answer:'''
        
        request = f'''Есть следующая информация
                    ---------------------
                    {retrieved_chunk}
                    ---------------------
                    Ответь на вопрос используя только информацию выше
                    Вопрос: {question}
                    Ответ:'''

        result = self.llm(prompt=request, **self.generationKwargs)

        print(I.tolist()[0])
        print(retrieved_chunk)
        print("Answer ------")
        print(result["choices"][0]["text"])
        print("Answer ------")


    def mainModelComputeRequest(self, request : str) -> str:

        request = f'''Ответь на вопрос: {request} '''
        
        result = self.llm(prompt=request, **self.generationKwargs)
        return result["choices"][0]["text"]

    def mainModelSetContextFile(self, path : str) -> None:
        pass

    def mainModelDownload(self, name : str) -> None:
        pass
