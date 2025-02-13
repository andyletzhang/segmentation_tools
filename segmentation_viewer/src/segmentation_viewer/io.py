import re

import numpy as np
from nd2 import ND2File
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QValidator
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from tifffile import TiffFile

from segmentation_tools.io import read_nd2, read_nd2_shape, read_tif, read_tif_shape


def read_image_file(file_path, progress_bar=None, image_shape=None, **progress_kwargs):
    if progress_bar is None:

        def progress_bar(x, **kwargs):
            return x

    if file_path.endswith('.nd2'):
        file = ND2File(file_path)
        shape = read_nd2_shape(file)  # (T, P, Z, C, Y, X)
        placeholder = read_nd2(file)  # Load the ND2 file
    else:
        file = TiffFile(file_path)
        shape, order = read_tif_shape(file)
        axes_map = {axis: i for i, axis in enumerate(reversed(order))}
        shape = tuple(shape[axes_map[axis]] for axis in 'TPZCYX')
        placeholder = read_tif(file)  # Load the TIFF file

    if image_shape is not None:
        if image_shape == 'all':
            t_bounds, p_bounds, z_bounds, c_bounds = (slice(None), slice(None), slice(None), slice(None))
        else:
            t_bounds, p_bounds, z_bounds, c_bounds = image_shape

    else:
        shape_dialog = ShapeDialog(shape)  # Prompt user to select ranges to import for each dimension
        if shape_dialog.exec_() == QDialog.Accepted:
            try:
                out = shape_dialog.get_selected_ranges()
                if out is None:
                    file.close()
                    return None
                else:
                    t_bounds, p_bounds, z_bounds, c_bounds = out
            except ValueError as e:
                print(f'Error: {e}')
        else:
            file.close()
            return None

    sliced = placeholder[np.ix_(t_bounds, p_bounds, z_bounds)]  # index the P, T, and Z dimensions
    img_shape = (shape[4], shape[5], len(c_bounds))  # output image shape (Y, X, C)
    # Assuming `sliced_array` is the array of functions
    result_array = np.empty((*sliced.shape, *img_shape), dtype=np.uint16)  # Initialize output array

    # Read the data from the file
    for idx, get_img in progress_bar(np.ndenumerate(sliced), length=sliced.size, **progress_kwargs):
        img = get_img()
        if img.ndim == 2:  # mono
            img = img[np.newaxis]
        result_array[idx] = img.transpose(1, 2, 0)[..., c_bounds]  # Call the function and store the result
    file.close()
    # remove either P or T dimension
    if result_array.shape[0] == 1:
        result_array = np.squeeze(result_array, axis=0)
    elif result_array.shape[1] == 1:
        result_array = np.squeeze(result_array, axis=1)
    return result_array


class ShapeDialog(QDialog):
    def __init__(self, shape, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Select dimension ranges')
        self.shape = shape
        self.setMinimumWidth(300)
        self.subset_ranges = {}

        # Main layout
        main_layout = QVBoxLayout()

        # Form layout for dimension inputs
        form_layout = QFormLayout()
        self.line_edits = {}
        dimensions = ['T', 'P', 'Z', 'C']

        for i, dim in enumerate(dimensions):
            range_edit = QLineEdit()
            range_edit.setPlaceholderText('All')
            range_edit.setText('')

            # Keep reference to the line edit for later
            self.line_edits[dim] = range_edit

            # Create label with range information
            label = f'{dim} (0 - {self.shape[i] - 1})'
            range_edit.setValidator(RangeStringValidator(self.shape[i] - 1, self))

            # Add row to form layout
            form_layout.addRow(label, range_edit)

        # Add form layout to main layout
        main_layout.addLayout(form_layout)

        # Add buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton('OK')
        cancel_button = QPushButton('Cancel')
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)

        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        self.setLayout(main_layout)

        # Set focus to this window
        self.activateWindow()
        self.raise_()

    def parse_range(self, range_str, max_value):
        """Parse a range string into a list of indices."""
        # Handle empty string or "All" case
        range_str = range_str.strip()
        if not range_str or range_str.lower() == 'all':
            return list(range(max_value))
        else:
            return range_string_to_list(range_str)

    def get_selected_ranges(self):
        """Get the selected ranges for each dimension."""
        ranges = {}
        try:
            for dim, line_edit in self.line_edits.items():
                dim_idx = ['T', 'P', 'Z', 'C'].index(dim)
                max_value = self.shape[dim_idx]
                indices = self.parse_range(line_edit.text(), max_value)
                ranges[dim] = indices

            # Convert to slice objects or lists based on whether the selection is contiguous
            slices = [np.array(ranges[dim]) for dim in ['T', 'P', 'Z', 'C']]

            if len(slices[0]) > 1 and len(slices[1]) > 1:
                QMessageBox.warning(self, 'Invalid Input', 'Currently, either P or T dimension can be loaded, but not both.')
                return None

            return tuple(slices)

        except ValueError as e:
            QMessageBox.warning(self, 'Invalid Input', str(e))
            return None


class ExportWizard(QDialog):
    def __init__(self, dataframe, total_rows, parent=None, root_path=''):
        super().__init__(parent)
        self.setWindowTitle('Export Wizard')
        self.setMinimumWidth(300)
        self.setFixedHeight(300)
        self.resize(400, 300)
        self.attributes = dataframe.columns.tolist()
        self.preview_rows = 3
        self.data = dataframe.iloc[: self.preview_rows]
        self.checked_attributes = []
        self.save_path = ''
        self.minimumColumnWidth = 60

        # Main layout
        main_layout = QVBoxLayout(self)

        # Save As section
        save_layout = QHBoxLayout()
        save_label = QLabel('Save As:')
        self.save_input = QLineEdit(self, text=root_path)
        browse_button = QPushButton('Browse...')
        browse_button.clicked.connect(self.browse_save_location)
        save_layout.addWidget(save_label)
        save_layout.addWidget(self.save_input)
        save_layout.addWidget(browse_button)
        main_layout.addLayout(save_layout)

        # CSV Preview section
        preview_label = QLabel('CSV Preview:')
        main_layout.addWidget(preview_label)
        self.table_preview = QTableWidget(self.preview_rows + 2, len(self.attributes), self)
        self.populate_table_preview()
        self.table_preview.resizeColumnsToContents()
        for col in range(len(self.attributes)):
            column_width = max(self.table_preview.columnWidth(col) + 10, self.minimumColumnWidth)
            self.table_preview.setColumnWidth(col, column_width)
        main_layout.addWidget(self.table_preview)
        n_rows_label = QLabel(f'Total rows: {total_rows}')
        main_layout.addWidget(n_rows_label)

        # Confirm and Cancel buttons
        button_layout = QHBoxLayout()
        confirm_button = QPushButton('Confirm')
        cancel_button = QPushButton('Cancel')
        self.save_input.returnPressed.connect(confirm_button.click)
        confirm_button.clicked.connect(self.confirm)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(confirm_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)

    def browse_save_location(self):
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save As', '', 'CSV Files (*.csv);;All Files (*)')
        if file_path:
            self.save_input.setText(file_path)

    def populate_table_preview(self):
        self.table_preview.horizontalHeader().hide()  # Hide default headers
        self.table_preview.setColumnCount(len(self.attributes))

        self.checkboxes = {}  # Store references to checkboxes for later access
        # create headers with checkboxes
        for col, attribute in enumerate(self.attributes):
            header_widget = QWidget()
            header_layout = QHBoxLayout(header_widget)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Add a checkbox to the header
            checkbox = QCheckBox(attribute)
            checkbox.setChecked(True)  # Default: selected
            checkbox.clicked.connect(lambda state, col=col: self.toggle_column_visibility(col, state))
            self.checkboxes[attribute] = checkbox  # Store reference to checkbox for later access
            header_layout.addWidget(checkbox)

            # Set the widget to the header
            self.table_preview.setCellWidget(0, col, header_widget)  # Column-index base `wiring`
        # Fill the remaining rows with placeholder data
        self.table_preview.setVerticalHeaderItem(0, QTableWidgetItem(''))
        for row in range(self.preview_rows):
            self.table_preview.setVerticalHeaderItem(row + 1, QTableWidgetItem(f'{row}'))
            for col in range(len(self.attributes)):
                try:
                    cell_str = f'{self.data.iloc[row, col]:.4g}'  # Limit to 4 significant digits
                    cell_item = QTableWidgetItem(cell_str)  # Add sample data
                    cell_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)  # right-align numbers
                except ValueError:
                    cell_str = str(self.data.iloc[row, col])  # Fallback to string
                    cell_item = QTableWidgetItem(cell_str)  # Add sample data
                    cell_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)  # left-align strings

                cell_item.setFlags(cell_item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Disable editing
                self.table_preview.setItem(row + 1, col, cell_item)

        self.table_preview.setVerticalHeaderItem(self.preview_rows + 1, QTableWidgetItem('...'))
        for col in range(len(self.attributes)):
            cell_item = QTableWidgetItem('...')
            cell_item.setFlags(cell_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            cell_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table_preview.setItem(self.preview_rows + 1, col, cell_item)

    def toggle_column_visibility(self, col, state):
        """Gray out or restore a column based on checkbox state."""
        for row in range(1, self.table_preview.rowCount()):  # Skip header row
            cell_item = self.table_preview.item(row, col)
            if state:  # Checkbox checked
                cell_item.setForeground(Qt.GlobalColor.white)  # Restore text color
            else:  # Checkbox unchecked
                cell_item.setForeground(Qt.GlobalColor.gray)  # Gray out text

    def confirm(self):
        # Validate save path
        self.save_path = self.save_input.text().strip()
        if not self.save_path:
            QMessageBox.warning(self, 'Validation Error', 'Please specify a save location.')
            return

        # Collect selected attributes
        self.checked_attributes = [key for key, checkbox in self.checkboxes.items() if checkbox.isChecked()]
        if not self.checked_attributes:
            QMessageBox.warning(self, 'Validation Error', 'Please select at least one column.')
            return

        # Close dialog with success
        self.accept()


class RangeStringValidator(QValidator):
    def __init__(self, max_value, parent=None):
        """
        Validator for integer sequences and ranges.
        Accepts formats like: "1", "1, 3", "1-5", "1, 3-5, 7, 9-11"

        Args:
            max_value (int): Maximum allowed integer value
            parent: Parent QObject
        """
        super().__init__(parent)
        self.max_value = max_value

        # Regular expression for basic format validation
        # Matches patterns like: digit(s) or digit(s)-digit(s), separated by commas
        self.format_regex = re.compile(r'^(\d+(-\d+)?)(,\s*\d+(-\d+)?)*$')

    def validate(self, input_str: str, pos: int) -> tuple[QValidator.State, str, int]:
        """
        Validate the input string.

        Returns:
            Tuple of (ValidationState, input_string, position)
        """
        # Empty string is intermediate
        if not input_str:
            return (QValidator.State.Intermediate, input_str, pos)

        # Check for invalid characters early
        if not re.match(r'^[\d,\s-]*$', input_str):
            return (QValidator.State.Invalid, input_str, pos)

        # Create a whitespace-free version just for validation
        # but keep original string intact
        check_str = ''.join(input_str.split())

        # Allow intermediate states while typing
        if check_str.endswith(',') or check_str.endswith('-'):
            return (QValidator.State.Intermediate, input_str, pos)

        # Check basic format
        if not self.format_regex.match(check_str):
            return (QValidator.State.Intermediate, input_str, pos)

        try:
            # Split into individual parts, preserving original whitespace
            parts = [p.strip() for p in input_str.split(',')]

            for part in parts:
                if '-' in part:  # Range of integers
                    start_str, end_str = part.split('-')
                    # Allow intermediate state while typing the end of range
                    if not end_str:
                        return (QValidator.State.Intermediate, input_str, pos)

                    start = int(start_str.strip())
                    end = int(end_str.strip())

                    if start > self.max_value or end > self.max_value:
                        return (QValidator.State.Invalid, input_str, pos)
                else:
                    # Handle single number, if it's not empty
                    if part.strip():
                        num = int(part.strip())
                        if num > self.max_value:
                            return (QValidator.State.Invalid, input_str, pos)

            return (QValidator.State.Acceptable, input_str, pos)

        except ValueError:
            return (QValidator.State.Intermediate, input_str, pos)

    def fixup(self, input_str: str) -> str:
        """
        Try to fix up the input string to make it valid.
        Standardizes spacing after commas and removes extra whitespace.
        """
        # Split by comma and clean each part
        parts = [part.strip() for part in input_str.split(',')]
        # Remove empty parts and join with standard spacing
        return ', '.join(part for part in parts if part)


def range_string_to_list(range_str):
    """
    Parse the input string into a list of integers.
    Accepts input in the form of "1-10, 15, 17-18".

    Returns:
        list: List of integers from the parsed input

    Raises:
        ValueError: If the input format is invalid
    """
    indices = []
    try:
        # Split by comma and handle each part
        parts = [p.strip() for p in range_str.split(',')]
        for part in parts:
            if '-' in part:
                # Handle range (e.g., "1-5")
                start, end = map(int, part.split('-'))
                if start > end:
                    # reversed range
                    indices.extend(reversed(range(end, start + 1)))
                else:
                    indices.extend(range(start, end + 1))
            else:
                # Handle single number
                num = int(part)
                indices.append(num)
        return indices
    except ValueError as e:
        raise ValueError(f'Invalid range format: {e}')
