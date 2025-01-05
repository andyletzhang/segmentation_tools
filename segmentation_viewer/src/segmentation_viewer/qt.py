from PyQt6.QtWidgets import QComboBox
from PyQt6.QtCore import pyqtSignal

class CustomComboBox(QComboBox):
    dropdownOpened=pyqtSignal()
    def showPopup(self):
        self.dropdownOpened.emit()
        super().showPopup()  # Call the original showPopup method