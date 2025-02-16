from huggingface_hub import hf_hub_download, snapshot_download
from llama_cpp import Llama
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import numpy as np
import faiss
import os

import json


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

    embedTextProcessFunctionProgress = None
    embedTextProcessFunctionStopFlag = False
    embedTextProcessFunctionFinish = None

    textGenerationCallbackFunction = None
    textGenerationFunctionStopFlag = False
    textGenerationFunctionFinish = None

    def LoadEmbeddingModelFromFile(self, path : str = "./models/multilingual-e5-large-instruct_q8_0.gguf", nCtx=4000, nThreads=54, nGPULayers=0) -> None:
        try:
            self.llmEmbeding = LlamaCppEmbeddings(
                model_path=path,
                n_ctx=nCtx,  # Context length to use
                n_threads=nThreads,            # Number of CPU threads to use
                n_gpu_layers=nGPULayers        # Number of model layers to offload to GPU
            )
            self.usedEmbeddingModel = path.split("/")[-1].split(".")[0]
        except Exception as e:
            raise RuntimeError("Can't instantiate gguf model " + str(e))

    def LoadAnswerModelFromFile(self, path : str = "./models/ggml-model-Q8_0.gguf", nCtx=4000, nThreads=54, nGPULayers=0) -> None:
        try:
            self.llm = Llama(
                model_path=path,
                n_ctx=nCtx,  # Context length to use
                n_threads=nThreads,            # Number of CPU threads to use
                n_gpu_layers=nGPULayers        # Number of model layers to offload to GPU
            )
        except Exception as e:
            raise RuntimeError("Can't instantiate gguf model " + str(e))
    
    def LoadGenerationKwargs(self, maxTokens : int = 2000, stop : list[str] = ["</s>"], echo : bool = False, topK = 1) -> None:
        self.generationKwargs = {
            "max_tokens":2000,
            "stop":["</s>"],
            "echo":False, # Echo the prompt in the output
            "top_k":1, # This is essentiallys greedy decoding, since the model will always return the highest-probability token. Set this value > 1 for sampling decoding
            "stream":True
        }

    def SplitText(self, text, textDocumentName = "document", chunkSize : int = 600, chunkOverlap : int = 100) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
        self.splittedTextForIndex = splitter.split_text(text)
        self.textDocumentName = textDocumentName
        return self.splittedTextForIndex

    def SetSplitTextsProcessFunction(self, func):
        self.embedTextProcessFunctionProgress = func
    
    def EmbedTexts(self) -> None:
    
        if self.splittedTextForIndex is None:
            raise RuntimeError("No splitted text is setted")

        if self.llmEmbeding is None:
            raise RuntimeError("No embedding model is setted")
        
        #asyncio.run(self.EmbedCycle)
        textsEmbeds = self.EmbedCycle()

        if textsEmbeds is None:
            return
        
        d = textsEmbeds.shape[1]

        if self.embedTextProcessFunctionProgress is not None:
            self.embedTextProcessFunctionProgress(0.0)

        if self.embedTextProcessFunctionFinish is not None:
                    self.embedTextProcessFunctionFinish()

        self.faissRAGIndex = faiss.IndexFlatL2(d)
        self.faissRAGIndex.add(textsEmbeds)
        return

        
    
    def EmbedCycle(self) -> np.array:

        totalCount = len(self.splittedTextForIndex)
        valuePerStep = 1.0 / float(totalCount)
        currentProgress = 0.0
        textsEmbeds = []

        for text in self.splittedTextForIndex:
            if self.embedTextProcessFunctionStopFlag == True:

                self.embedTextProcessFunctionStopFlag = False

                if self.embedTextProcessFunctionProgress is not None:
                    self.embedTextProcessFunctionProgress(0.0)

                if self.embedTextProcessFunctionFinish is not None:
                    self.embedTextProcessFunctionFinish()
                return

            # Because of embed queue sometimes incorrectly parse model results in it
            currentProgress += valuePerStep

            if self.embedTextProcessFunctionProgress is not None:
                self.embedTextProcessFunctionProgress(currentProgress)
            
            resultArray = self.llmEmbeding.client.embed(text)
            textsEmbeds.append(resultArray)
        
        textsEmbeds = np.array(textsEmbeds)
        self.embedTextProcessFunctionStopFlag = False
        
        return textsEmbeds
    

    def SaveTextAndEmbededVectorStorage(self, path : str = "./vector_db/") -> None:
        if self.faissRAGIndex is None:
            raise RuntimeError("No index is setted")
        
        path += '/' + self.textDocumentName + "_" + self.usedEmbeddingModel

        try:
            os.makedirs(path + "/index")
        except FileExistsError:
            print("Save directory already exists")
        except PermissionError:
            print("No permission to create save dir")

        
        try:
            os.makedirs(path + "/text")
        except FileExistsError:
            print("Save directory already exists")
        except PermissionError:
            print("No permission to create save dir")

        try:
            with open(path + "/index/index", "w"):
                faiss.write_index(self.faissRAGIndex, path + "/index/index")
            with open(path + "/text/data", "w") as file:
                file.write(json.dumps(self.splittedTextForIndex))
                file.close()

        except Exception as e:
            raise RuntimeError("Error on saving vector data base " + str(e))

        
        
    def LoadTextAndEmbededVectorStorage(self, path : str = "./vector_db/") -> None:
        try:
            self.faissRAGIndex = faiss.read_index(path + "/index/index")
            with open(path + "/text/data", "r") as file:
                self.splittedTextForIndex = json.loads(file.read())
        
        except Exception as e:
            raise RuntimeError("Error on loading vector data base " + str(e))

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
        
    def ComputePrompt(self, question : str, preset : str = "{0}",  k = 2, extend = 0) -> str:
        ''' Computte a prompt from preset and fill it with appropriate text chunks '''
        try:
            embededQuestion = self.EmbedQuestion(question)
        except Exception as e:
            raise RuntimeError("Error on embedding question " + str(e))
        
        chunks = ["None"]
        try:
            chunks = self.FindChunks(embededQuestion=embededQuestion, k=k, extend=extend)
        except Exception as e: 
            raise RuntimeError("Error on search chunks " + str(e))
        
        result = preset.format(question, chunks)
        print (result)
        
        return result

    def asdf() -> None:
        pass

        question = "Who is Valakas"
        question = "Кто такой Валакас?"
        question = "Типы данных, не относящиеся к ОО расширениям, но позволяющие хранить неатомарные значения?"
        question_embedding = np.array([self.llmEmbeding.client.embed(question)])

        D, I = self.search(question_embedding, k=2)

        retrieved_chunk = [self.splittedText[i] for i in I.tolist()[0]]
        
        # request = f
        
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


    def ComputeRequest(self, prompt : str) -> None:
        """Returns all the text results from llm"""
        if self.llm is None:
            raise RuntimeError ("No answer model is setted")
        
        output = self.llm.create_completion(prompt=prompt, **self.generationKwargs)
        # output = self.llm(prompt=prompt, **self.generationKwargs)
        
        #resultsTexts = [choice["text"] for choice in result["choices"]]
        result = ""

        for out in output: 
            if self.textGenerationFunctionStopFlag == True:

                if self.textGenerationCallbackFunction is not None:
                    self.textGenerationCallbackFunction(result)

                if self.textGenerationFunctionFinish is not None:
                    self.textGenerationFunctionFinish()

                self.textGenerationFunctionStopFlag = False

                return

            print(out['choices'][0]['text'], end='\n')
            result += out['choices'][0]['text']
            if self.textGenerationCallbackFunction is not None:
                self.textGenerationCallbackFunction(result)
        
        if self.textGenerationFunctionFinish is not None:
                    self.textGenerationFunctionFinish()
                    
            


    def mainModelSetContextFile(self, path : str) -> None:
        pass

    def DownloadModel(modelName : str = "intfloat/multilingual-e5-large-instruct") -> None:
        filePath = snapshot_download(modelName, local_dir="./models/downloaded/" + modelName.split('/')[1], local_dir_use_symlinks=False, revision="main", tqdm_class=None)
