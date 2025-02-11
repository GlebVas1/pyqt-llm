#import model_downloader
#import model

import main_ui_controller
import sys
from PyQt5 import QtWidgets

# model_downloader.downloadModel()

# thisModel = model.mainModel()
# thisModel.LoadAnswerModelFromFile()
# thisModel.LoadEmbeddingModelFromFile()
# thisModel.LoadGenerationKwargs()

# text = ""
# with open("test.txt", 'r') as file:
#     text = file.read()

# thisModel.SplitText(text, "test", 600, 100)
# thisModel.EmbedTexts()
# thisModel.SaveTextAndEmbededVectorStorage()

# prompt = thisModel.ComputePrompt("Что ты можешь сказать о старшем брате?", extend=1)
# print(thisModel.ComputeRequest(prompt))

#print(thisModel.mainModelComputeRequest("Кто такой пушкин"))

class App(QtWidgets.QMainWindow, main_ui_controller.Controller):
    def __init__(self):
        super().__init__()

app = QtWidgets.QApplication(sys.argv)

frame = App()
frame.show()

frame.AddToListViewMessages('''
*Italic*    **Bold**
# Heading 1
## Heading 2

[Link](http://a.com)

* List
* List
* List

- [x] @mentions, #refs, [links](), **formatting**, and <del>tags</del> supported
- [x] list syntax required (any unordered or ordered list supported)
- [x] this is a complete item
- [ ] this is an incomplete item

First Header | Second Header
------------ | -------------
Content from cell 1 | Content from cell 2
Content in the first column | Content in the second column
''',
0)
frame.AddToListViewMessages('''
                            AAAA
                            BBB
                            CCC
                            ''', 1)

frame.AddToListViewMessages('''
                            AAAA
                            BBB
                            CCC
                            ''', 0)

frame.AddToListViewMessages('''
                            AAAA
                            BBB
                            CCC
                            ''', 0)

frame.AddToListViewMessages('''
                            AAAA
                            BBB
                            CCC
                            ''', 1)

app.exec()