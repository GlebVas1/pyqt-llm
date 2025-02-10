import model_downloader
import model

# model_downloader.downloadModel()

thisModel = model.mainModel()
thisModel.LoadAnswerModelFromFile()
thisModel.LoadEmbeddingModelFromFile()
thisModel.LoadGenerationKwargs()

text = ""
with open("test.txt", 'r') as file:
    text = file.read()

thisModel.SplitText(text, "test", 600, 100)
thisModel.EmbedTexts()
thisModel.SaveTextAndEmbededVectorStorage()

prompt = thisModel.ComputePrompt("О чем эта статья?", extend=1)
print(thisModel.ComputeRequest(prompt))

#print(thisModel.mainModelComputeRequest("Кто такой пушкин"))