from PyQt6.QtWidgets import QComboBox
from PyQt6.QtGui import QAction
from PyQt6.QtCore import pyqtSignal

class CustomComboBox(QComboBox):
    '''Custom QComboBox that emits a signal when the dropdown is opened'''
    dropdownOpened=pyqtSignal()
    def showPopup(self):
        self.dropdownOpened.emit()
        super().showPopup()  # Call the original showPopup method

def create_action(name, func, parent=None, shortcut=None):
    action=QAction(name, parent)
    action.triggered.connect(func)
    if shortcut is not None:
        action.setShortcut(shortcut)
    return action