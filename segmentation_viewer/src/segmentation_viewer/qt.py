from PyQt6.QtWidgets import QComboBox, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt6.QtGui import QAction, QMouseEvent
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from superqt import QRangeSlider

def create_action(name, func, parent=None, shortcut=None):
    action=QAction(name, parent)
    action.triggered.connect(func)
    if shortcut is not None:
        action.setShortcut(shortcut)
    return action

class CustomComboBox(QComboBox):
    '''Custom QComboBox that emits a signal when the dropdown is opened'''
    dropdownOpened=pyqtSignal()
    def showPopup(self):
        self.dropdownOpened.emit()
        super().showPopup()  # Call the original showPopup method

class FineScrubQRangeSlider(QRangeSlider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fine_scrubbing = False
        self.fine_scrub_factor = 0.1  # Adjust this to change sensitivity
        self.original_press_pos = None

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton:
            # Create a modified left-click event at the same position
            self.fine_scrubbing = True
            self.original_press_pos = event.position()
            
            modified_event = QMouseEvent(
                event.type(),
                event.position(),
                event.globalPosition(),
                Qt.MouseButton.LeftButton,  # Convert right click to left
                Qt.MouseButton.LeftButton,  # Active buttons
                event.modifiers()
            )
            super().mousePressEvent(modified_event)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.RightButton and self.fine_scrubbing:
            # Create a modified left-button release event
            modified_event = QMouseEvent(
                event.type(),
                event.position(),
                event.globalPosition(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,  # No buttons pressed after release
                event.modifiers()
            )
            self.fine_scrubbing = False
            self.original_press_pos = None
            super().mouseReleaseEvent(modified_event)
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.fine_scrubbing and self.original_press_pos is not None:
            # Calculate the scaled movement
            delta = event.position() - self.original_press_pos
            scaled_delta = delta * self.fine_scrub_factor
            
            # Create new position that applies the scaled movement to the original position
            fine_pos = self.original_press_pos + scaled_delta
            
            # Create modified move event with scaled position
            modified_event = QMouseEvent(
                event.type(),
                QPointF(fine_pos),
                event.globalPosition(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                event.modifiers()
            )
            super().mouseMoveEvent(modified_event)
        else:
            super().mouseMoveEvent(event)
