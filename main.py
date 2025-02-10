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

prompt = thisModel.ComputePrompt("Перечисли все DLC с кратким описанием", extend=2)
print(thisModel.ComputeRequest(prompt))

#print(thisModel.mainModelComputeRequest("Кто такой пушкин"))