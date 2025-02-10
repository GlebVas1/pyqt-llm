from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
import os

class PromptPatterns:
    languageDict = {}

    def AddPromptTemplate():
        pass

class Parameters():
    splitChunkSize = 800
    splitChunkOverlap = 100
    embeddingGuffModel = False
    llmEmbeding = None
    llm = None

class mainModel(Parameters):

    usedAnswerModel = None
    usedEmbeddingModel = None
    faissRAGIndex = None
    splittedTextForIndex = None
    textDocumentName = None

    generationKwargs = None


    def InitializeModel(self, name : str = "1") -> None:
        pass

    def LoadEmbeddingModelFromFile(self, path : str = "./models/multilingual-e5-large-instruct_q8_0.gguf") -> None:
        self.llmEmbeding = LlamaCppEmbeddings(
            model_path=path,
            n_ctx=4000,  # Context length to use
            n_threads=54,            # Number of CPU threads to use
            n_gpu_layers=0        # Number of model layers to offload to GPU
        )
        self.usedEmbeddingModel = path.split("/")[-1].split(".")[0]

    def LoadGenerationKwargs(self) -> None:
        self.generationKwargs = {
            "max_tokens":2000,
            "stop":["</s>"],
            "echo":False, # Echo the prompt in the output
            "top_k":1 # This is essentiallys greedy decoding, since the model will always return the highest-probability token. Set this value > 1 for sampling decoding
        }

    def LoadAnswerModelFromFile(self, path : str = "./models/Llama-3.1-Tulu-3-8B-Q8_0.gguf") -> None:
        self.llm = Llama(
            model_path=path,
            n_ctx=4000,  # Context length to use
            n_threads=54,            # Number of CPU threads to use
            n_gpu_layers=0        # Number of model layers to offload to GPU
        )
    

    def SplitText(self, text, textDocumentName = "document", chunkSize : int = 600, chunkOverlap : int = 100) -> list[str]:

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
        self.splittedTextForIndex = splitter.split_text(text)
        self.textDocumentName = textDocumentName
        return self.splittedTextForIndex

    def EmbedTexts(self) -> None:

        if self.splittedTextForIndex is None:
            raise RuntimeError("No splitted text is setted")

        if self.llmEmbeding is None:
            raise RuntimeError("No embedding model is setted")
        
        textsEmbeds = []

        for text in self.splittedTextForIndex:
            # Because of embed queue sometimes incorrectly parse model results in it
            resultArray = self.llmEmbeding.client.embed(text)
            textsEmbeds.append(resultArray)
        
        textsEmbeds = np.array(textsEmbeds)

        d = textsEmbeds.shape[1]

        self.faissRAGIndex = faiss.IndexFlatL2(d)
        self.faissRAGIndex.add(textsEmbeds)

    def SaveTextAndEmbededVectorStorage(self, path : str = "./vector_db/") -> None:
        if self.faissRAGIndex is None:
            raise RuntimeError("No index is setted")
        
        path += self.textDocumentName + "_" + self.usedEmbeddingModel

        try:
            os.makedirs(path + "/index")
        except FileExistsError:
            print("Save directory already exists")
        except PermissionError:
            print("No permission to create save dir")


        with open(path + "/index/index", "w"):
            faiss.write_index(self.faissRAGIndex, path + "/index/index")

        try:
            os.makedirs(path + "/text")
        except FileExistsError:
            print("Save directory already exists")
        except PermissionError:
            print("No permission to create save dir")

        with open(path + "/text/data", "w") as file:
            file.write(str(self.splittedTextForIndex))
        


    def EmbedQuestion(self, question : str) -> np.array:
        if self.llmEmbeding is not None:
            return np.array([self.llmEmbeding.client.embed(question)])
        else:
            raise RuntimeError("No embedding model is setted")
        
    def FindChunks(self, embededQuestion : np.array, k : int = 2, extend = 0) -> list[str]:
        if self.faissRAGIndex is None:
            raise RuntimeError("No index is setted")

        if self.splittedTextForIndex is None:
            raise RuntimeError("No texts for indexing")
        
        try:
            D, I = self.faissRAGIndex.search(embededQuestion, k=k)

            ids = list(I.tolist()[0])

            finalIds = []

            for i in ids:
                for j in range(-extend, extend + 1):
            
                    if i + j >= 0 and i + j < len(self.splittedTextForIndex):
                        finalIds.append(i + j)
            
            chunks = [self.splittedTextForIndex[i] for i in finalIds]
            return chunks

        except Exception as e:
            raise RuntimeError("Error while searching: " + str(e))
        
    def ComputePrompt(self, question : str, k = 2, extend = 0) -> str:
        embededQuestion = self.EmbedQuestion(question)
        chunks = self.FindChunks(embededQuestion=embededQuestion, k=k, extend=extend)
        result = f'''Есть следующая информация
                    ---------------------
                    {chunks}
                    ---------------------
                    Ответь на вопрос используя только информацию выше
                    Вопрос: {question}
                    Ответ:'''
        return result

    def asdf() -> None:
        pass

        question = "Who is Valakas"
        question = "Кто такой Валакас?"
        question = "Типы данных, не относящиеся к ОО расширениям, но позволяющие хранить неатомарные значения?"
        question_embedding = np.array([self.llmEmbeding.client.embed(question)])

        D, I = self.search(question_embedding, k=2)

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


    def ComputeRequest(self, prompt : str) -> list[str]:
        """Returns all the text results from llm"""
        print(self.generationKwargs)
        result = self.llm(prompt=prompt, **self.generationKwargs)

        resultsTexts = [choice["text"] for choice in result["choices"]]
        return resultsTexts

    def mainModelSetContextFile(self, path : str) -> None:
        pass

    def mainModelDownload(self, name : str) -> None:
        pass
