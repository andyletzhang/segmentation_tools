from PyQt6.QtWidgets import QComboBox, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QColorDialog, QWidget, QGridLayout
from PyQt6.QtGui import QAction, QMouseEvent, QColor, QFont, QDoubleValidator
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
        submit_layout=QHBoxLayout()
        self.button_confirm = QPushButton("Confirm", self)
        self.button_cancel = QPushButton("Cancel", self)
        submit_layout.addWidget(self.button_confirm)
        submit_layout.addWidget(self.button_cancel)

        layout.addLayout(submit_layout)
        
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
        
class OverlaySettingsDialog(QDialog):
    settings_applied = pyqtSignal(dict)
    def __init__(self, parent: QWidget = None):
        """
        Initialize the dialog. Inherit initial settings from the parent.
        """
        super().__init__(parent)
        self.setWindowTitle("Overlay Settings")

        # Inherit initial values from the parent
        self.selected_cell_color = QColor(getattr(parent, "selected_cell_color"))
        self.selected_cell_alpha = str(getattr(parent, "selected_cell_alpha"))
        self.masks_alpha = str(getattr(parent, "masks_alpha"))
        self.outlines_color = QColor(getattr(parent, "outlines_color"))
        self.outlines_alpha = str(getattr(parent, "outlines_alpha"))

        # Main dialog layout
        dialog_layout = QVBoxLayout(self)

        # Grid layout for rows
        settings_grid = QGridLayout()
        dialog_layout.addLayout(settings_grid)

        # Create bold font
        bold_font = QFont()
        bold_font.setBold(True)

        # Add header labels directly to grid
        gui_label = QLabel("GUI Element")
        color_label = QLabel("Color")
        alpha_label = QLabel("Alpha")
        
        for label in [gui_label, color_label, alpha_label]:
            label.setFont(bold_font)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        settings_grid.addWidget(gui_label, 0, 0)
        settings_grid.addWidget(color_label, 0, 1)
        settings_grid.addWidget(alpha_label, 0, 2)

        # Selected cell row
        self.selected_cell_color_swatch, self.selected_cell_alpha_line=self.add_color_alpha_row(
            settings_grid, 1, 
            "Selected Cell", 
            self.selected_cell_color, 
            self.selected_cell_alpha, 
            self.change_selected_cell_color
        )

        # Masks row (no color picker)
        self.masks_alpha_line=self.add_alpha_row(settings_grid, 2, "Masks", self.masks_alpha)

        # Outlines row
        self.outlines_color_swatch, self.outlines_alpha_line=self.add_color_alpha_row(
            settings_grid, 3, 
            "Outlines", 
            self.outlines_color, 
            self.outlines_alpha, 
            self.change_outlines_color,
        )

        # Set column stretches to control alignment
        settings_grid.setColumnStretch(0, 2)  # GUI Element column
        settings_grid.setColumnStretch(1, 1)  # Color column
        settings_grid.setColumnStretch(2, 1)  # Alpha column

        # Confirm and Cancel buttons
        submit_layout = QHBoxLayout()
        self.button_ok = QPushButton("OK", self)
        self.button_cancel = QPushButton("Cancel", self)
        self.button_apply = QPushButton("Apply", self)
        submit_layout.addWidget(self.button_ok)
        submit_layout.addWidget(self.button_cancel)
        submit_layout.addWidget(self.button_apply)

        dialog_layout.addLayout(submit_layout)

        # Connect buttons
        self.button_ok.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)
        self.button_apply.clicked.connect(self.apply_settings)

    def add_color_alpha_row(self, layout, row, label_text, initial_color, initial_alpha, color_change_callback):
        """
        Add a row with a color picker and an alpha input to the grid layout.
        """
        label = QLabel(label_text)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(label, row, 0)

        color_button=QPushButton()
        color_button.setFixedSize(40, 20)
        color_button.setStyleSheet(f"background-color: {initial_color.name()}; border: 1px solid black;")
        color_button.clicked.connect(color_change_callback)
        layout.addWidget(color_button, row, 1, Qt.AlignmentFlag.AlignCenter)

        alpha_input = QLineEdit()
        alpha_input.setValidator(QDoubleValidator(0.0, 1.0, 3))
        alpha_input.setFixedWidth(50)
        alpha_input.setPlaceholderText(initial_alpha)
        alpha_input.setText(initial_alpha)
        layout.addWidget(alpha_input, row, 2, Qt.AlignmentFlag.AlignCenter)

        return color_button, alpha_input

    def add_alpha_row(self, layout, row, label_text, initial_alpha):
        """
        Add a row with only an alpha input to the grid layout (no color picker).
        """
        label = QLabel(label_text)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(label, row, 0)

        spacer = QLabel()  # Empty spacer for alignment
        layout.addWidget(spacer, row, 1, Qt.AlignmentFlag.AlignCenter)

        alpha_input = QLineEdit()
        alpha_input.setValidator(QDoubleValidator(0.0, 1.0, 3))
        alpha_input.setFixedWidth(50)
        alpha_input.setPlaceholderText(initial_alpha)
        alpha_input.setText(initial_alpha)
        layout.addWidget(alpha_input, row, 2, Qt.AlignmentFlag.AlignCenter)

        return alpha_input

    def change_selected_cell_color(self):
        """
        Open a QColorDialog to change the selected cell color.
        """
        color = QColorDialog.getColor(self.selected_cell_color, self, "Choose Selected Cell Color")
        if color.isValid():
            self.selected_cell_color = color
            self.selected_cell_color_swatch.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")

    def change_outlines_color(self):
        """
        Open a QColorDialog to change the outlines color.
        """
        color = QColorDialog.getColor(self.outlines_color, self, "Choose Outlines Color")
        if color.isValid():
            self.outlines_color = color
            self.outlines_color_swatch.setStyleSheet(f"background-color: {color.name()}; border: 1px solid black;")

    def apply_settings(self):
        """
        Apply and store the settings.
        """
        self.settings_applied.emit(self.get_settings())

    def get_settings(self):
        """
        Returns the selected settings.
        """
        if not self.selected_cell_alpha_line.text():
            selected_cell_alpha=self.selected_cell_alpha
        else:
            selected_cell_alpha=float(self.selected_cell_alpha_line.text())
        
        if not self.masks_alpha_line.text():
            masks_alpha=self.masks_alpha
        else:
            masks_alpha=float(self.masks_alpha_line.text())

        if not self.outlines_alpha_line.text():
            outlines_alpha=self.outlines_alpha
        else:
            outlines_alpha=float(self.outlines_alpha_line.text())
        
        return {'selected_cell_color': self.selected_cell_color.name(),
            'selected_cell_alpha': selected_cell_alpha,
            'masks_alpha': masks_alpha,
            'outlines_color': self.outlines_color.name(),
            'outlines_alpha': outlines_alpha
        }