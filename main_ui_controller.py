import mainui
import model
import os

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QSize, QRunnable, QThreadPool, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QGraphicsDropShadowEffect, QFileDialog
from functools import partial

from dialog_item import CustomDialogWidget

import psutil as ps 

import presets as prs

import threading


class Controller(mainui.Ui_MainWindow):

    LLMModel = model.mainModel()
    threadpool = QThreadPool()
    
    allMesagesWidgets = []

    # https://stackoverflow.com/questions/25733142/qwidgetrepaint-recursive-repaint-detected-when-updating-progress-bar
    # https://stackoverflow.com/questions/45556440/pyqt-emit-signal-from-threading-thread
    embedProgressSignal = pyqtSignal(float)
    embedButtonsSignal = pyqtSignal()

    embedThread = None


    answerStringSignal = pyqtSignal(str)
    answerButtonsSignal = pyqtSignal()


    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.InitializeComboBoxes()
        self.InitializeActions()
        self.InitializeChatField()
        self.InitializeThreadsSpinBoxes()
        self.InitializePresets()
        self.AddShadows()

        os.environ["QT_ENABLE_HIGHDPI_SCALING"]   = "0"
        os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
        os.environ["QT_SCALE_FACTOR"]             = "0"

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
        self.VectorDataBaseStopPushButton.clicked.connect(self.StopEmbedThread)
        
        self.PresetComboBox.currentTextChanged.connect(self.LoadPreset)
        self.embedProgressSignal.connect(self.ChangeEmbedProgressBar)
        self.embedButtonsSignal.connect(self.ChangeVectorDataBaseProcessButtons)
        
        self.answerStringSignal.connect(self.AnswerCallbackFrom)
        self.answerButtonsSignal.connect(self.ChangePromptButtons)

        self.PromptStopPushButon.clicked.connect(self.StopAnswerThread)
        self.VectorDataBaseStopPushButton.setEnabled(False)
        self.PromptStopPushButon.setEnabled(False)
    
    def InitializePresets(self):
        for name, pr in prs.presets.items():
            self.PresetComboBox.addItem(name)

        self.PresetComboBox.setCurrentIndex(0)
        self.PresetTextEdit.setText(prs.presets[self.PresetComboBox.currentText()])

    def LoadPreset(self):
        self.PresetTextEdit.setText(prs.presets[self.PresetComboBox.currentText()])
    
    def InitializeThreadsSpinBoxes(self):
        self.AnswerModelThreadsSpinBox.setMaximum(ps.cpu_count())
        self.EmbeddingModelThreadsSpinBox.setMaximum(ps.cpu_count())

        self.AnswerModelThreadsSpinBox.setValue(ps.cpu_count() - 2)
        self.EmbeddingModelThreadsSpinBox.setValue(ps.cpu_count() - 2)


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

    def ChangeMessage(self, ind : int, text : str):
        self.allMesagesWidgets[ind].ChangeMessage(text)


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

        self.allMesagesWidgets.append(messageOut)
        
        #item.setTextAlignment(Qt.AlignmentFlag.AlignLeft if type == 1 else Qt.AlignmentFlag.AlignRight)
        
        self.DialogListWidget.addItem(item)
        self.DialogListWidget.setItemWidget(item, messageOut)
        self.DialogListWidget.scrollToBottom()
        self.DialogListWidget.show()
        

    def SendPrompt(self):
        #self.AddToListViewMessages(self.PromptTextEdit.toPlainText(), type=0)
        #self.AddToListViewMessages(self.PromptTextEdit.toPlainText(), type=1)

        #return

        # computedPrompt = "None"
        # try:
        #     computedPrompt = self.LLMModel.ComputePrompt(self.PromptTextEdit.toPlainText(),
        #                                                 preset=self.PresetTextEdit.toPlainText(),
        #                                                 k=self.VectorDataBaseSearchKSpinBox.value(),
        #                                                 extend=self.VectorDataBaseExtendSpinBox.value())
        # except Exception as e:
        #     self.ShowMessageBox(str(e))
        #     return
        
        # self.LLMModel.LoadGenerationKwargs(
        #     maxTokens=self.GenerationSettingsMaxTokensSpinBox.value(),
        #     topK=self.GenerationSettingsTopKSpinBox.value(),
        #     echo=self.GenerationSettingsEchoCheckBox.isChecked()
        # )
        # result = None
        self.LLMModel.textGenerationCallbackFunction = self.AnswerCallbackFromThread
        self.LLMModel.textGenerationFunctionFinish = self.ChangePromptButtonsFromThread
        self.LLMModel.LoadGenerationKwargs(
            maxTokens=self.GenerationSettingsMaxTokensSpinBox.value(),
            topK=self.GenerationSettingsTopKSpinBox.value(),
            echo=self.GenerationSettingsEchoCheckBox.isChecked()
        )

        self.AddToListViewMessages(self.PromptTextEdit.toPlainText(), type=0)

        computedPrompt = "What is red wine?"
        self.AddToListViewMessages("", type=1)

        try: 
            answerThread = threading.Thread(target=partial(self.LLMModel.ComputeRequest, computedPrompt))
            answerThread.start()
            
        except Exception as e:
            self.ShowMessageBox(str(e))

        self.PromptSendPushButton.setEnabled(False)
        self.PromptStopPushButon.setEnabled(True)

        self.PromptTextEdit.setText("")
        

    def AnswerCallbackFromThread(self, str):
        self.answerStringSignal.emit(str)

    def AnswerCallbackFrom(self, str):
        self.ChangeMessage(-1, str)

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


    def ChangeEmbedProgressBar(self, value : float):
        self.VectorDataBaseProgressBar.setValue(int(100.0 * value))
        print("Val " + str(value))

    def ChangeEmbedProgressBarFromThread(self, value : float = 0.0):
        self.embedProgressSignal.emit(value)
        
    def ChangeVectorDataBaseProcessButtons(self):
        self.VectorDataBaseProcessPushButton.setEnabled(True)
        self.VectorDataBaseStopPushButton.setEnabled(False)


    def ChangeVectorDataBaseProcessButtonsFromThread(self):
        self.embedButtonsSignal.emit()

    def EmbedSplittedText(self):
        self.LLMModel.embedTextProcessFunctionProgress = self.ChangeEmbedProgressBarFromThread
        self.LLMModel.embedTextProcessFunctionFinish = self.ChangeVectorDataBaseProcessButtonsFromThread

        try:
           embedThread = threading.Thread(target=self.LLMModel.EmbedTexts, daemon=False)
           embedThread.start()

        except Exception as e:
            self.ShowMessageBox(str(e))
            return
        
        self.VectorDataBaseProcessPushButton.setEnabled(False)
        self.VectorDataBaseStopPushButton.setEnabled(True)


    def StopEmbedThread(self):
        self.LLMModel.embedTextProcessFunctionStopFlag = True

    def StopAnswerThread(self):
        self.LLMModel.textGenerationFunctionStopFlag = True

    def ChangePromptButtons(self):
        self.PromptSendPushButton.setEnabled(True)
        self.PromptStopPushButon.setEnabled(False)

    def ChangePromptButtonsFromThread(self):
        self.answerButtonsSignal.emit()






    
        
