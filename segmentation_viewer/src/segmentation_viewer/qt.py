from PyQt6.QtWidgets import QComboBox, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt6.QtGui import QAction, QMouseEvent
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from superqt import QRangeSlider
from segmentation_viewer.io import RangeStringValidator, range_string_to_list

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

class SubstackDialog(QDialog):
    def __init__(self, array_length, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Input Substack Frames")
        
        # Layout and widgets
        layout = QVBoxLayout(self)
        self.label = QLabel("Input substack frames:", self)
        layout.addWidget(self.label)
        
        self.line_edit = QLineEdit(self)
        self.line_edit.setPlaceholderText("e.g. 1-10, 15, 17-18")
        
        # Set the validator to the QLineEdit
        self.line_edit.setValidator(RangeStringValidator(array_length-1, self))
        
        layout.addWidget(self.line_edit)
        
        # Confirm and Cancel buttons
        self.button_confirm = QPushButton("Confirm", self)
        self.button_cancel = QPushButton("Cancel", self)
        layout.addWidget(self.button_confirm)
        layout.addWidget(self.button_cancel)
        
        # Connect buttons
        self.button_confirm.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)

    def get_input(self):
        """
        Returns the text entered by the user in the QLineEdit.
        """
        try:
            return range_string_to_list(self.line_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Invalid input. Please enter a valid range of frames.")
            return None