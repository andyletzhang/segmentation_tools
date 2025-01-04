
from nd2 import ND2File
from segmentation_tools.io import read_nd2, read_nd2_shape, read_tif, read_tif_shape
from tifffile import TiffFile
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QLineEdit, QPushButton, QHBoxLayout
from pathlib import Path
import numpy as np

def read_image_file(file_path, progress_bar=None, **progress_kwargs):
    if progress_bar is None:
        progress_bar = lambda x, **kwargs: x
    if file_path.endswith('.nd2'):
        file=ND2File(file_path)
        shape=read_nd2_shape(file) # (T, Z, C, Y, X)
        placeholder=read_nd2(file) # Load the ND2 file
    else:
        file=TiffFile(file_path)
        shape, order=read_tif_shape(file)
        axes_map = {axis: i for i, axis in enumerate(reversed(order))}
        shape=tuple(shape[axes_map[axis]] for axis in 'TZCYX')
        placeholder=read_tif(file) # Load the TIFF file

    shape_dialog = ShapeDialog(shape) # Prompt user to select ranges to import for each dimension
    if shape_dialog.exec_() == QDialog.Accepted:
        try:
            t_bounds, z_bounds, c_bounds=shape_dialog.get_selected_ranges()
        except ValueError as e:
            print(f"Error: {e}")
    else:
        return None
    
    sliced=placeholder[np.ix_(t_bounds, z_bounds)] # index the T and Z dimensions
    img_shape=(shape[3], shape[4], len(c_bounds)) # output image shape (Y, X, C)

    # Assuming `sliced_array` is the array of functions
    result_array = np.empty((*sliced.shape, *img_shape), dtype=np.uint16)  # Initialize output array

    # Read the data from the file
    for idx, func in progress_bar(np.ndenumerate(sliced), length=sliced.size, **progress_kwargs):
        result_array[idx] = func()[c_bounds].transpose(1,2,0)  # Call the function and store the result
    file.close()
    return result_array

class ShapeDialog(QDialog):
    def __init__(self, shape, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select dimension ranges")
        self.shape = shape
        self.setMinimumWidth(300)
        self.subset_ranges = {}
        
        # Main layout
        main_layout = QVBoxLayout()
        
        # Form layout for dimension inputs
        form_layout = QFormLayout()
        self.line_edits = {}
        dimensions = ['T', 'Z', 'C']
        
        for i, dim in enumerate(dimensions):
            range_edit = QLineEdit()
            range_edit.setPlaceholderText("All")
            range_edit.setText("")
            
            # Keep reference to the line edit for later
            self.line_edits[dim] = range_edit
            
            # Create label with range information
            label = f"{dim} (0 - {self.shape[i] - 1})"
            
            # Add row to form layout
            form_layout.addRow(label, range_edit)
        
        # Add form layout to main layout
        main_layout.addLayout(form_layout)
        
        # Add buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
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
        if not range_str or range_str.lower() == "all":
            return list(range(max_value))
            
        indices = []
        try:
            # Split by comma and handle each part
            parts = [p.strip() for p in range_str.split(',')]
            for part in parts:
                if '-' in part:
                    # Handle range (e.g., "1-5")
                    start, end = map(int, part.split('-'))
                    if start < 0 or end >= max_value:
                        raise ValueError(f"Values must be between 0 and {max_value-1}")
                    if start>end:
                        # reversed range
                        indices.extend(reversed(range(end, start + 1)))
                    else:
                        indices.extend(range(start, end + 1))
                else:
                    # Handle single number
                    num = int(part)
                    if num < 0 or num >= max_value:
                        raise ValueError(f"Values must be between 0 and {max_value-1}")
                    indices.append(num)
            return indices
        except ValueError as e:
            raise ValueError(f"Invalid range format: {e}")
    
    def get_selected_ranges(self):
        """Get the selected ranges for each dimension."""
        ranges = {}
        try:
            for dim, line_edit in self.line_edits.items():
                dim_idx = ['T', 'Z', 'C'].index(dim)
                max_value = self.shape[dim_idx]
                indices = self.parse_range(line_edit.text(), max_value)
                ranges[dim] = indices
            
            # Convert to slice objects or lists based on whether the selection is contiguous
            slices = [np.array(ranges[dim]) for dim in ['T', 'Z', 'C']]
            
            return tuple(slices)
        
        except ValueError as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid Input", str(e))
            return None