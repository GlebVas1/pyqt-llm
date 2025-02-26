from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QSize, QRect
from PyQt5.QtWidgets import QMessageBox, QGraphicsDropShadowEffect, QFileDialog

def StyleSheetByTypeOfTheMessage(type : int) -> str:
    if type == 0:
        return '''
                border-radius : 7px;
                background-color : rgb(250, 250, 250); \n
                color : black;
                border-color : rgb(230, 230, 230);
                padding-left : 10px;
                padding-top : 10px;
                '''
    else:
        return '''
                border-radius : 7px;
                background-color : rgb(90, 90, 90); \n
                color : white;
                border-color : rgb(100, 100, 100);
                padding-left : 10px;
                padding-top : 10px;
                '''
    

class CustomDialogWidget(QtWidgets.QWidget):

    def Initialize(self, text : str, type : int = 0) ->None:
        # 0 -- client
        # 1 -- llm 

        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(75)


        self.textField = QtWidgets.QTextEdit(self)
        self.textField.setObjectName("message")
        self.textField.setStyleSheet(StyleSheetByTypeOfTheMessage(type))
        self.textField.setFont(font)

        self.textField.setMarkdown(text)
        
        self.textField.updateGeometry()
        self.textField.document().adjustSize()
        self.textField.show()
        
        self.textField.setReadOnly(True)
        
        print(self.textField.document().size().height())

        self.actualHeight = int(self.textField.document().size().height()) + 20
        self.actualWidth = max(int(self.textField.document().size().width()) + 20, self.width() * 2 // 3)

        self.textField.setMinimumHeight(self.actualHeight)
        self.textField.setMaximumHeight(self.actualHeight)

        self.textField.setMinimumWidth(self.actualWidth)
        self.textField.setMaximumWidth(self.actualWidth)

        self.setMinimumHeight(self.actualHeight)
        self.setMaximumHeight(self.actualHeight)
        self.setMinimumWidth(self.actualWidth)

        effect = QGraphicsDropShadowEffect()
        effect.setOffset(0, 0)
        effect.setBlurRadius(15)
        self.textField.setGraphicsEffect(effect)
        
        self.type = type

        self.show()

    def sizeHint(self):
        return QSize(self.width(), self.actualHeight)
    
    def Resize(self):
        
        self.textField.updateGeometry()
        self.textField.document().adjustSize()
        self.textField.show()
        print("New text field size")
        print(self.textField.document().size().height())
        self.actualHeight = int(self.textField.document().size().height()) + 20
        self.actualWidth = max(int(self.textField.document().size().width()) + 20, self.width() * 2 // 3)

        self.textField.setMinimumWidth(self.actualWidth)
        self.textField.setMaximumWidth(self.actualWidth)


        self.textField.setMinimumHeight(self.actualHeight)
        self.textField.setMaximumHeight(self.actualHeight)

        self.setMinimumWidth(self.actualWidth)
        self.setMinimumHeight(self.actualHeight)
        self.setMaximumHeight(self.actualHeight)

        horizontalOffset = 0 if self.type == 1 else self.width() - self.actualWidth
        self.textField.setGeometry(QRect(horizontalOffset, 0, self.actualWidth, self.actualHeight))

    def ChangeMessage(self, text : str):
        self.textField.setMarkdown(text)
        self.Resize()
        
