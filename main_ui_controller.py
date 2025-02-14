import mainui
import model
import os

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QMessageBox, QGraphicsDropShadowEffect, QFileDialog
from functools import partial

from dialog_item import CustomDialogWidget

class Controller(mainui.Ui_MainWindow):
    LLMModel = model.mainModel()


    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.InitializeComboBoxes()
        self.InitializeActions()
        self.InitializeChatField()
        self.AddShadows()

    def AddShadows(self):
        frames = [self.AnswerModelFrame,
         self.DialogFrame,
         self.EmbeddingModelFrame,
         self.GenerationSettingsFrame,
         self.LanguageFrame,
         self.LogoFrame,
         self.ModelDownloadingFrame,
         self.PromptTextFrame, 
         self.PresetFrame, 
         self.TextProcessingFrame,
         self.VectorDataBaseFrame
        ]

        for frame in frames:
            effect = QGraphicsDropShadowEffect()
            effect.setOffset(0, 0)
            effect.setBlurRadius(15)
            frame.setGraphicsEffect(effect)


    def ShowMessageBox(self, text : str) -> None:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(text)
        msg.setWindowTitle("Error")
        msg.exec_()

    def OpenFileDialog(self) -> str:
        file_dialog = QFileDialog(self)
        file_dialog.setWindowTitle("Open File")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setViewMode(QFileDialog.ViewMode.Detail)

        selectedFile = "None"

        if file_dialog.exec():
            selectedFile = file_dialog.selectedFiles()[0]
    
        return selectedFile
        
    def InitializeChatField(self):
        self.DialogListWidget.setSpacing(10)
        self.DialogListWidget.setContentsMargins(5,5,5,5)

    
    def InitializeComboBoxes(self) -> None:
        try:
            os.mkdir("./models/embedding")
        except FileExistsError:
            print("Models embedding dir already exists")
        except:
            print("Can not create or find models embedding dir")
            return
        
        try:
            os.mkdir("./models/answer")
        except FileExistsError:
            print("Models answer dir already exists")
        except:
            print("Can not create or find models answer dir")
            return

        embedModelsNames = os.listdir("./models/embedding")
        self.EmbeddingModelComboBox.addItems(embedModelsNames)

        answerModelsNames = os.listdir("./models/answer")
        self.AnswerModelComboBox.addItems(answerModelsNames)

    def InitializeActions(self):
        self.AnswerModelLoadPushButton.clicked.connect(self.LoadAnswerModelFromFile)
        self.EmbeddingModelLoadPushButton.clicked.connect(self.LoadEmbeddingModelFromFile)
        self.PromptSendPushButton.clicked.connect(self.SendPrompt)
        self.TextProcessingLoadPushButton.clicked.connect(self.LoadTextFile)
        self.VectorDataBaseProcessPushButton.clicked.connect(self.EmbedSplittedText)
    
    def LoadAnswerModelFromFile(self) -> None:
        try:
            self.LLMModel.LoadAnswerModelFromFile(
                path="./models/answer/" + self.AnswerModelComboBox.currentText(),
                nCtx=self.AnswerModelContextTokensSpinBox.value(),
                nThreads=self.AnswerModelThreadsSpinBox.value(),
                nGPULayers=self.AnswerModelGPULayersSpinBox.value()
            )
        except RuntimeError as e:
            self.ShowMessageBox(str(e))

    def LoadEmbeddingModelFromFile(self) -> None:
        try:
            self.LLMModel.LoadEmbeddingModelFromFile(
                path="./models/embedding/" + self.EmbeddingModelComboBox.currentText(),
                nCtx=self.EmbeddingModelContextTokensSpinBox.value(),
                nThreads=self.EmbeddingModelThreadsSpinBox.value(),
                nGPULayers=self.EmbeddingModelGPULayersSpinBox.value()
            )
        except RuntimeError as e:
            self.ShowMessageBox(str(e))

    def AddToListViewMessages(self, message : str, type : int = 0):
        
        # type 1 = answer
        # type 0 = prompt

        messageOut = CustomDialogWidget()
        messageOut.Initialize(message, type)

        item = QtWidgets.QListWidgetItem(self.DialogListWidget) 
        
        # it doesn't update even after show()
        # item.setSizeHint(messageOut.sizeHint())

        messageOut.setFixedWidth(self.DialogListWidget.width() - 40)

        messageOut.Resize()

        item.setSizeHint(messageOut.sizeHint())

        #item.setTextAlignment(Qt.AlignmentFlag.AlignLeft if type == 1 else Qt.AlignmentFlag.AlignRight)
        
        self.DialogListWidget.addItem(item)
        self.DialogListWidget.setItemWidget(item, messageOut)
        self.DialogListWidget.scrollToBottom()
        self.DialogListWidget.show()
        

    def SendPrompt(self):
        #self.AddToListViewMessages(self.PromptTextEdit.toPlainText(), type=0)
        #self.AddToListViewMessages(self.PromptTextEdit.toPlainText(), type=1)

        #return
        computedPrompt = "None"
        try:
            computedPrompt = self.LLMModel.ComputePrompt(self.PromptTextEdit.toPlainText(),
                                                        k=self.VectorDataBaseSearchKSpinBox.value(),
                                                        extend=self.VectorDataBaseExtendSpinBox.value())
        except Exception as e:
            self.ShowMessageBox(str(e))
            return
        
        self.LLMModel.LoadGenerationKwargs(
            maxTokens=self.GenerationSettingsMaxTokensSpinBox.value(),
            topK=self.GenerationSettingsTopKSpinBox.value(),
            echo=self.GenerationSettingsEchoCheckBox.isChecked()
        )
        result = None
        try: 
            result = self.LLMModel.ComputeRequest(computedPrompt)
            
        except Exception as e:
            self.ShowMessageBox(str(e))
            return

        self.AddToListViewMessages(self.PromptTextEdit.toPlainText(), type=0)
        self.PromptTextEdit.setText("")
        self.AddToListViewMessages(result[0], type=1)



    def LoadTextFile(self):
        filePath = self.OpenFileDialog()
        if filePath == "None":
            self.ShowMessageBox("No file was selected")
            return
        with open(filePath, "r") as file:
            try:
                text = file.read()
                self.LLMModel.SplitText(text,
                                        filePath.split('/')[-1],
                                        chunkSize=self.TextProcessingChunkSizeSpinBox.value(),
                                        chunkOverlap=self.TextProcessingChunkOverlapSpinBox.value())
            except Exception as e:
                self.ShowMessageBox("Can not split file " + str(e))
        
        print("File was splitted")

    def ChangeEmbedProgressBar(self, value : float = 0.0):
        self.VectorDataBaseProgressBar.setValue(int(100.0 * value))

    def EmbedSplittedText(self):
        self.LLMModel.splitTextProcessFunction = self.ChangeEmbedProgressBar
        try:
            self.LLMModel.EmbedTexts()
        except Exception as e:
            self.ShowMessageBox(str(e))





    
        
