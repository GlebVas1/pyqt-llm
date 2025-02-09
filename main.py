import model_downloader
import model

# model_downloader.downloadModel()

thisModel = model.mainModel()
thisModel.loadModelfromFile()
text = ""
with open("test.txt", 'r') as file:
    text = file.read()

thisModel.mainModelSplitAndEmbedText(text)

#print(thisModel.mainModelComputeRequest("Кто такой пушкин"))