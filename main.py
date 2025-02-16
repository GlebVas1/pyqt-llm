import main_ui_controller
import sys
from PyQt5 import QtWidgets


class App(QtWidgets.QMainWindow, main_ui_controller.Controller):
    def __init__(self):
        super().__init__()

app = QtWidgets.QApplication(sys.argv)

frame = App()
frame.show()

app.exec()