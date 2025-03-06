from PyQt6.QtCore import QPointF, Qt, pyqtSignal, QTimer, QObject
from PyQt6.QtGui import QColor, QIntValidator, QDoubleValidator, QFont, QMouseEvent, QUndoStack, QUndoCommand
from PyQt6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QSpacerItem,
    QSizePolicy,
    QUndoView,
    QMainWindow,
)
from superqt import QRangeSlider, QDoubleRangeSlider
from segmentation_viewer.io import RangeStringValidator, range_string_to_list


class CustomComboBox(QComboBox):
    """Custom QComboBox that emits a signal when the dropdown is opened"""

    dropdownOpened = pyqtSignal(QComboBox)  # Signal emitted when the dropdown is opened

    def showPopup(self):
        self.dropdownOpened.emit(self)
        super().showPopup()  # Call the original showPopup method

    def changeToText(self, text):
        """Set the current text of the combobox without triggering the currentIndexChanged signal"""
        # add item to the combobox if it doesn't exist
        if self.findText(text) == -1:
            self.addItem(text)
        super().setCurrentText(text)


class FineScrubber():
    def __init__(self, *args, scale=1, **kwargs):
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
                event.modifiers(),
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
                event.modifiers(),
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
                event.modifiers(),
            )
            super().mouseMoveEvent(modified_event)
        else:
            super().mouseMoveEvent(event)

    def rescale(self, min_val: float, max_val: float, step: float) -> None:
        """
        Rescale the slider's min/max, step size to the provided values.
        """
        self._singleStep = step
        self._pageStep = step * 10
        self.setRange(min_val, max_val)
    
class FineDoubleRangeSlider(QDoubleRangeSlider, FineScrubber):
    def __init__(self, *args, scale=1, **kwargs):
        super().__init__(*args, scale=scale, **kwargs)
        self._singleStep=scale
        self._pageStep=scale*10

class FineRangeSlider(QRangeSlider, FineScrubber):
    pass


def labeled_LUT_slider(slider_name=None, mode='int', decimals=2, default_range=(0, 65535), parent=None, digit_width=5):
    labels_and_slider = QHBoxLayout()
    labels_and_slider.setSpacing(2)
    if slider_name is not None:
        slider_label = QLabel(slider_name)
        labels_and_slider.addWidget(slider_label)

    if mode == 'int':
        slider = FineRangeSlider(orientation=Qt.Orientation.Horizontal, parent=parent)
        validator=QIntValidator(*default_range)
    elif mode == 'float':
        slider = FineDoubleRangeSlider(orientation=Qt.Orientation.Horizontal, parent=parent)
        validator=QDoubleValidator(*default_range, decimals)
    else:
        raise ValueError(f"Invalid slider mode: {mode}")
    slider.setRange(*default_range)
    slider.setValue(default_range)

    range_labels = [QLineEdit(str(val)) for val in slider.value()]
    for label in range_labels:
        label.setFixedWidth(digit_width * 6)
        label.setAlignment(Qt.AlignTop)
        label.setValidator(validator)
        label.setStyleSheet("""
            QLineEdit {
                border: none;
                background: transparent;
                padding: 0;
            }
        """)
    range_labels[0].setAlignment(Qt.AlignRight)

    def format_label(label):
        if mode == 'int':
            return str(int(label))
        else:
            return str(round(label, decimals))

    def format_text(text):
        if mode == 'int':
            return int(text)
        else:
            return float(text)

    # Connect QLineEdit changes to update the slider value
    def update_min_slider_from_edit():
        min_val = format_text(range_labels[0].text())
        max_val = format_text(range_labels[1].text())

        if min_val < slider.minimum():
            slider.setMinimum(min_val)
        elif min_val > max_val:
            min_val = max_val
            range_labels[0].setText(str(min_val))
        slider.setValue((min_val, max_val))

    def update_max_slider_from_edit():
        min_val = format_text(range_labels[0].text())
        max_val = format_text(range_labels[1].text())

        if max_val > slider.maximum():
            slider.setMaximum(max_val)
        elif max_val < min_val:
            max_val = min_val
            range_labels[1].setText(str(max_val))
        slider.setValue((min_val, max_val))

    range_labels[0].editingFinished.connect(update_min_slider_from_edit)
    range_labels[1].editingFinished.connect(update_max_slider_from_edit)

    # Connect slider value changes to update the QLineEdit text
    def update_edits_from_slider():
        min_val, max_val = slider.value()
        min_val=format_label(min_val)
        max_val=format_label(max_val)
        range_labels[0].setText(min_val)
        range_labels[1].setText(max_val)

    slider.valueChanged.connect(update_edits_from_slider)

    labels_and_slider.addWidget(range_labels[0])
    labels_and_slider.addSpacerItem(QSpacerItem(10, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed))
    labels_and_slider.addWidget(slider)
    labels_and_slider.addSpacerItem(QSpacerItem(10, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed))
    labels_and_slider.addWidget(range_labels[1])

    return labels_and_slider, slider, range_labels

class SubstackDialog(QDialog):
    def __init__(self, array_length, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Input Substack Frames')

        # Layout and widgets
        layout = QVBoxLayout(self)
        self.label = QLabel('Input substack frames:', self)
        layout.addWidget(self.label)

        self.line_edit = QLineEdit(self)
        self.line_edit.setPlaceholderText('e.g. 1-10, 15, 17-18')

        # Set the validator to the QLineEdit
        self.line_edit.setValidator(RangeStringValidator(array_length - 1, self))

        layout.addWidget(self.line_edit)

        # Confirm and Cancel buttons
        submit_layout = QHBoxLayout()
        self.button_confirm = QPushButton('Confirm', self)
        self.button_cancel = QPushButton('Cancel', self)
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
            QMessageBox.warning(self, 'Invalid Input', 'Invalid input. Please enter a valid range of frames.')
            return None

class OverlaySettingsDialog(QDialog):
    settings_applied = pyqtSignal(dict)

    def __init__(self, parent: QWidget = None):
        """
        Initialize the dialog. Inherit initial settings from the parent.
        """
        super().__init__(parent)
        self.setWindowTitle('Overlay Settings')

        # Inherit initial values from the parent
        self.selected_cell_color = QColor(getattr(parent, 'selected_cell_color'))
        self.selected_cell_alpha = str(getattr(parent, 'selected_cell_alpha'))
        self.masks_alpha = str(getattr(parent, 'masks_alpha'))
        self.outlines_color = QColor(getattr(parent, 'outlines_color'))
        self.outlines_alpha = str(getattr(parent, 'outlines_alpha'))

        # Main dialog layout
        dialog_layout = QVBoxLayout(self)

        # Grid layout for rows
        settings_grid = QGridLayout()
        dialog_layout.addLayout(settings_grid)

        # Create bold font
        bold_font = QFont()
        bold_font.setBold(True)

        # Add header labels directly to grid
        gui_label = QLabel('GUI Element')
        color_label = QLabel('Color')
        alpha_label = QLabel('Alpha')

        for label in [gui_label, color_label, alpha_label]:
            label.setFont(bold_font)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        settings_grid.addWidget(gui_label, 0, 0)
        settings_grid.addWidget(color_label, 0, 1)
        settings_grid.addWidget(alpha_label, 0, 2)

        # Selected cell row
        self.selected_cell_color_swatch, self.selected_cell_alpha_line = self.add_color_alpha_row(
            settings_grid, 1, 'Selected Cell', self.selected_cell_color, self.selected_cell_alpha, self.change_selected_cell_color
        )

        # Masks row (no color picker)
        self.masks_alpha_line = self.add_alpha_row(settings_grid, 2, 'Masks', self.masks_alpha)

        # Outlines row
        self.outlines_color_swatch, self.outlines_alpha_line = self.add_color_alpha_row(
            settings_grid, 3, 'Outlines', self.outlines_color, self.outlines_alpha, self.change_outlines_color
        )

        # Set column stretches to control alignment
        settings_grid.setColumnStretch(0, 2)  # GUI Element column
        settings_grid.setColumnStretch(1, 1)  # Color column
        settings_grid.setColumnStretch(2, 1)  # Alpha column

        # Confirm and Cancel buttons
        submit_layout = QHBoxLayout()
        self.button_ok = QPushButton('OK', self)
        self.button_cancel = QPushButton('Cancel', self)
        self.button_apply = QPushButton('Apply', self)
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

        color_button = QPushButton()
        color_button.setFixedSize(40, 20)
        color_button.setStyleSheet(f'background-color: {initial_color.name()}; border: 1px solid black;')
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
        color = QColorDialog.getColor(self.selected_cell_color, self, 'Choose Selected Cell Color')
        if color.isValid():
            self.selected_cell_color = color
            self.selected_cell_color_swatch.setStyleSheet(f'background-color: {color.name()}; border: 1px solid black;')

    def change_outlines_color(self):
        """
        Open a QColorDialog to change the outlines color.
        """
        color = QColorDialog.getColor(self.outlines_color, self, 'Choose Outlines Color')
        if color.isValid():
            self.outlines_color = color
            self.outlines_color_swatch.setStyleSheet(f'background-color: {color.name()}; border: 1px solid black;')

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
            selected_cell_alpha = self.selected_cell_alpha
        else:
            selected_cell_alpha = float(self.selected_cell_alpha_line.text())

        if not self.masks_alpha_line.text():
            masks_alpha = self.masks_alpha
        else:
            masks_alpha = float(self.masks_alpha_line.text())

        if not self.outlines_alpha_line.text():
            outlines_alpha = self.outlines_alpha
        else:
            outlines_alpha = float(self.outlines_alpha_line.text())

        return {
            'selected_cell_color': self.selected_cell_color.name(),
            'selected_cell_alpha': selected_cell_alpha,
            'masks_alpha': masks_alpha,
            'outlines_color': self.outlines_color.name(),
            'outlines_alpha': outlines_alpha,
        }


def bordered(widget):
    border_wrapper = QWidget(objectName='bordered')
    border_layout = QVBoxLayout(border_wrapper)
    border_layout.setContentsMargins(0, 0, 0, 0)
    border_layout.addWidget(widget)
    return border_wrapper


class CollapsibleWidget(QWidget):
    def __init__(self, header_text, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setContentsMargins(0, 0, 0, 0)
        self.core_layout = QVBoxLayout()
        self.header_layout = QHBoxLayout()
        self.header_layout.setContentsMargins(0, 0, 0, 0)
        self.toggle_hidden = QToolButton(text=header_text)
        self.toggle_hidden.setCheckable(True)
        self.toggle_hidden.setChecked(True)
        self.toggle_hidden.setStyleSheet('QToolButton { border: none; }')
        self.toggle_hidden.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_hidden.setArrowType(Qt.DownArrow)
        self.toggle_hidden.toggled.connect(self.on_toggled)

        # Content Widget with items
        self.collapsing_widget = QWidget()
        self.collapsing_layout = QVBoxLayout(self.collapsing_widget)

        self.header_layout.addWidget(self.toggle_hidden)
        self.header_layout.addStretch()
        self.core_layout.addLayout(self.header_layout)
        self.core_layout.addWidget(self.collapsing_widget)
        self.setLayout(self.core_layout)

    def show_content(self):
        self.toggle_hidden.setChecked(True)

    def hide_content(self):
        self.toggle_hidden.setChecked(False)

    def addWidget(self, widget):
        self.collapsing_layout.addWidget(widget)

    def addLayout(self, layout):
        self.collapsing_layout.addLayout(layout)

    def addSpacerItem(self, spacer_item):
        self.collapsing_layout.addSpacerItem(spacer_item)

    def on_toggled(self, checked):
        if checked:
            self.toggle_hidden.setArrowType(Qt.DownArrow)
            self.collapsing_widget.show()
        else:
            self.toggle_hidden.setArrowType(Qt.RightArrow)
            self.collapsing_widget.hide()


class ChannelOrderDialog(QDialog):
    previewRequested=pyqtSignal(list)
    clearPreviewRequested=pyqtSignal()
    finished=pyqtSignal(list)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Channel Order')

        # Layout and widgets
        layout = QVBoxLayout(self)
        self.label = QLabel('Enter channel order:', self)
        layout.addWidget(self.label)

        self.line_edit = QLineEdit(self)
        self.line_edit.setValidator(RangeStringValidator(max_value=2, parent=self))
        self.line_edit.setPlaceholderText('e.g. 1, 2, 0')
        self.line_edit.textChanged.connect(self._check_input)

        layout.addWidget(self.line_edit)

        # Confirm and Cancel buttons
        submit_layout = QHBoxLayout()
        self.button_confirm = QPushButton('Confirm', self)
        self.button_cancel = QPushButton('Cancel', self)
        submit_layout.addWidget(self.button_cancel)
        submit_layout.addWidget(self.button_confirm)

        layout.addLayout(submit_layout)

        # Connect buttons
        self.button_confirm.clicked.connect(self.confirm_and_finish)
        self.button_cancel.clicked.connect(self.clear_and_reject)

    def _check_input(self):
        try:
            channel_order=self.get_input()
            if len(channel_order) == 3:
                self.previewRequested.emit(channel_order)
        except ValueError:
            self.clearPreviewRequested.emit()
        
    def clear_and_reject(self):
        self.clearPreviewRequested.emit()
        self.reject()

    def get_input(self):
        """
        Returns the text entered by the user in the QLineEdit.
        """
        return range_string_to_list(self.line_edit.text())
    
    def confirm_and_finish(self):
        try:
            channel_order = self.get_input()
            if len(channel_order) == 3:
                self.finished.emit(channel_order)
                self.accept()
            else:
                QMessageBox.warning(self, 'Invalid Input', f'Expected 3 channels, got {len(channel_order)}.')
        except ValueError:
            QMessageBox.warning(self, 'Invalid Input', f'Invalid input: {self.line_edit.text()}')
            return None

class LookupTableDialog(QDialog):
    valueChanged=pyqtSignal(list)
    def __init__(self, parent=None, options=[], initial_LUTs=[]):
        super().__init__(parent)
        self.setWindowTitle('Edit LUTs')

        # Layout and widgets
        layout=QVBoxLayout(self)
        dropdown_layout = QFormLayout()
        dropdown_layout.setSpacing(2)
        self.label = QLabel('Enter lookup tables:', self)
        layout.addWidget(self.label)

        red_label = QLabel('R:')
        green_label = QLabel('G:')
        blue_label = QLabel('B:')
        self.red_dropdown = QComboBox(self)
        self.green_dropdown = QComboBox(self)
        self.blue_dropdown = QComboBox(self)

        for dropdown, current_value in zip([self.red_dropdown, self.green_dropdown, self.blue_dropdown], initial_LUTs):
            dropdown.addItems(options)
            dropdown.setCurrentText(current_value)
            dropdown.currentIndexChanged.connect(self.send_preview)
        
        dropdown_layout.addRow(red_label, self.red_dropdown)
        dropdown_layout.addRow(green_label, self.green_dropdown)
        dropdown_layout.addRow(blue_label, self.blue_dropdown)

        submit_layout = QHBoxLayout()
        self.button_confirm = QPushButton('Confirm', self)
        self.button_cancel = QPushButton('Cancel', self)
        submit_layout.addStretch()
        submit_layout.addWidget(self.button_cancel)
        submit_layout.addWidget(self.button_confirm)

        layout.addLayout(dropdown_layout)
        layout.addLayout(submit_layout)
        
        # Connect buttons
        self.button_confirm.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)

    def send_preview(self):
        self.valueChanged.emit(self.get_input())

    def get_input(self):
        return [self.red_dropdown.currentText(), self.green_dropdown.currentText(), self.blue_dropdown.currentText()]

class FrameStackDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        # dialog to save either frame or stack
        self.setWindowTitle("Save Options")
        
        # Create custom button box
        layout=QHBoxLayout(self)
        save_stack_button = QPushButton("Save Stack")
        save_frame_button = QPushButton("Save Frame")
        
        # Add buttons to layout
        layout.addWidget(save_frame_button)
        layout.addWidget(save_stack_button)
        
        # Connect signals
        save_stack_button.clicked.connect(self.save_stack)
        save_frame_button.clicked.connect(self.save_frame)

        # Layout
        self.setLayout(layout)

    def save_stack(self):
        self.output = 'stack'
        self.done(QDialog.DialogCode.Accepted)

    def save_frame(self):
        self.output = 'frame'
        self.done(QDialog.DialogCode.Accepted)
        
    @staticmethod
    def get_choice(parent=None):
        dialog = FrameStackDialog(parent)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            return dialog.output
        else:
            return None

class UndoHistoryWindow(QMainWindow):
    """A child window that displays the undo stack history."""
    def __init__(self, stack, parent=None):
        super().__init__(parent)
        self.setWindowTitle("History")

        # Create central widget
        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)

        # Add QUndoView
        self.undo_view = QUndoView(stack)
        layout.addWidget(self.undo_view)

        self.setCentralWidget(central_widget)
        self.show()

class QueuedUndoStack(QUndoStack):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.push_queue = UndoStackManager(self)
        self.undo_queue = []  # Will contain 'undo' or 'redo' strings
        self.processing = False
        
    def queuedUndo(self):
        # If the last queued operation is a redo, cancel it out
        if self.undo_queue and self.undo_queue[-1] == 'redo':
            self.undo_queue.pop()
        else:
            # Otherwise, queue an undo
            self.undo_queue.append('undo')
        
        self.processQueue()
        
    def queuedRedo(self):
        # If the last queued operation is an undo, cancel it out
        if self.undo_queue and self.undo_queue[-1] == 'undo':
            self.undo_queue.pop()
        else:
            # Otherwise, queue a redo
            self.undo_queue.append('redo')
        
        self.processQueue()
    
    def processQueue(self):
        # Don't start processing if we're already processing or if the queue is empty
        if self.processing or not self.undo_queue:
            return
            
        self.processing = True
        operation = self.undo_queue.pop(0)  # Get and remove the first operation
        
        if operation == 'undo':
            super().undo()
        else:  # operation == 'redo'
            super().redo()
        
        self.processing = False
        
        if self.undo_queue:
            QTimer.singleShot(0, self.processQueue)

    # Override the original methods to use queued versions
    def undo(self):
        self.queuedUndo()
        
    def redo(self):
        self.queuedRedo()

    def _push(self, command):
        super().push(command)

    def push(self, command):
        self.push_queue.push(command)

class UndoStackManager(QObject):
    """Manages queued command execution for QUndoStack"""    
    def __init__(self, undo_stack: QUndoStack):
        super().__init__()
        self.undo_stack = undo_stack
        self.pending_commands = []
        self.is_executing = False
    
    def push(self, command: QUndoCommand):
        """Add command to queue and start processing if not already running"""
        self.pending_commands.append(command)
        
        if not self.is_executing:
            self._process_next_command()
    
    def _process_next_command(self):
        """Process the next command in the queue"""
        if not self.pending_commands:
            self.is_executing = False
            return
        
        self.is_executing = True
        command = self.pending_commands.pop(0)
        
        self.undo_stack._push(command)
        self._process_next_command()