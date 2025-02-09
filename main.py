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

thisModel.SplitText(text, 600, 100)
thisModel.EmbedTexts()
prompt = thisModel.ComputePrompt("Что сказал Иван Золо по поводу видео?")
print(thisModel.ComputeRequest(prompt))

#print(thisModel.mainModelComputeRequest("Кто такой пушкин"))