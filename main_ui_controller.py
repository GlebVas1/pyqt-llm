import mainui
import model
import os


from PyQt5.QtWidgets import QMessageBox

class Controller(mainui.Ui_MainWindow):
    LLMModel = model.mainModel()

    def ShowMessageBox(self, text : str) -> None:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(text)
        msg.exec_()
    
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

        with os.listdir("./models/embedding") as embedModelsNames:
            self.EmbeddingModelComboBox.addItems(embedModelsNames)

        with os.listdir("./models/answer") as answerModelsNames:
            self.AnswerModelComboBox.addItems(answerModelsNames)

    def InitializeActions(self):
        self.AnswerModelLoadPushButton.clicked.connect(self.LoadAnswerModelFromFile)
        self.EmbeddingModelLoadPushButton.clicked.connect(self.LoadEmbeddingModelFromFile)
        self.PromptSendPushButton.clicked.connect
    
    def LoadAnswerModelFromFile(self) -> None:
        try:
            self.LLMModel.LoadAnswerModelFromFile(
                path="./models/" + self.AnswerModelComboBox.currentText(),
                nCtx=self.AnswerModelContextTokensSpinBox.value(),
                nThreads=self.AnswerModelThreadsSpinBox.value(),
                nGPULayers=self.AnswerModelGPULayersSpinBox.value()
            )
        except RuntimeError as e:
            self.ShowMessageBox(str(e))

    def LoadEmbeddingModelFromFile(self) -> None:
        try:
            self.LLMModel.LoadEmbeddingModelFromFile(
                path="./models/" + self.EmbeddingModelComboBox.currentText(),
                nCtx=self.EmbeddingModelContextTokensSpinBox.value(),
                nThreads=self.EmbeddingModelThreadsSpinBox.value(),
                nGPULayers=self.EmbeddingModelGPULayersSpinBox.value()
            )
        except RuntimeError as e:
            self.ShowMessageBox(str(e))

    def SendPrompt(self):

        computedPrompt = "None"
        
        try:
            computedPrompt = self.LLMModel.ComputePrompt(self.PromptTextEdit.toPlainText(),
                                                        k=self.VectorDataBaseSearchKSpinBox.value(),
                                                        extend=self.VectorDataBaseExtendSpinBox.value())
        except Exception as e:
            self.ShowMessageBox(str(e))
            return
    
        try: 
            result = self.LLMModel.ComputeRequest(computedPrompt)

        except Exception as e:
            self.ShowMessageBox(str(e))

        





    
        
