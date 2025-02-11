from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QSize, QRect
from PyQt5.QtWidgets import QMessageBox, QGraphicsDropShadowEffect, QFileDialog

def StyleSheetByTypeOfTheMessage(type : int) -> str:
    if type == 0:
        return '''
                background-color : white; \n
                color : black;
                '''
    else:
        return '''
                background-color : rgb(120, 120, 120); \n
                color : white;
                '''
    

class CustomDialogWidget(QtWidgets.QWidget):

    def Initialize(self, text : str, type : int = 0) ->None:
        # 0 -- client
        # 1 -- llm 


        self.textField = QtWidgets.QTextEdit(self)
        self.textField.setObjectName("message")
        self.textField.setStyleSheet('''
                                    background-color : white; \n
                                    color : black;
                                    ''')

        self.textField.setMarkdown(text)
        self.textField.document().adjustSize()
        self.textField.updateGeometry()

        print(self.textField.document().size().height())

        self.actualHeight = int(self.textField.document().size().height()) + 40
        self.actualWidth = int(self.textField.document().size().width()) + 40

        self.textField.setMinimumHeight(self.actualHeight)
        self.textField.setMaximumHeight(self.actualHeight)

        self.textField.setMinimumWidth(self.actualWidth)
        self.textField.setMaximumWidth(self.actualWidth)

        self.setMinimumHeight(self.actualHeight)
        self.setMaximumHeight(self.actualHeight)
        self.setMinimumWidth(self.actualWidth)

        self.type = type

        self.show()

    def sizeHint(self):
        return QSize(self.width(), self.actualHeight)
    
    def Resize(self):
        horizontalOffset = 0 if self.type == 1 else self.width() - self.actualWidth
        self.textField.setGeometry(QRect(horizontalOffset, 0, self.actualWidth, self.actualHeight))
