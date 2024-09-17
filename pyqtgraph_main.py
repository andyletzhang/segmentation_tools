import sys
import numpy as np
import io
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QComboBox, QPushButton, QRadioButton,
    QVBoxLayout, QHBoxLayout, QCheckBox, QSpacerItem, QSizePolicy, QFileDialog,
    QLineEdit, QTextEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
import pyqtgraph as pg

from monolayer_tracking.segmented_comprehension import Image, TimeSeries
from monolayer_tracking import preprocessing

from tqdm import tqdm

dark_mode_style = """
    QWidget {
        background-color: #2e2e2e;
        color: #ffffff;
    }

    QMainWindow {
        background-color: #2e2e2e;
        border: 1px solid #1c1c1c;
    }

    QPushButton {
        background-color: #3e3e3e;
        color: #ffffff;
        border: 1px solid #555555;
        border-radius: 5px;
        padding: 5px;
    }

    QPushButton:hover {
        background-color: #4e4e4e;
    }

    QPushButton:pressed {
        background-color: #1c1c1c;
    }

    QLineEdit {
        background-color: #3e3e3e;
        color: #ffffff;
        border: 1px solid #555555;
    }

    QTextEdit {
        background-color: #3e3e3e;
        color: #ffffff;
        border: 1px solid #555555;
    }

    QMenuBar {
        background-color: #2e2e2e;
        color: #ffffff;
    }

    QMenuBar::item {
        background-color: #2e2e2e;
        color: #ffffff;
    }

    QMenuBar::item:selected {
        background-color: #4e4e4e;
    }

    QToolBar {
        background-color: #3e3e3e;
        border: 1px solid #1c1c1c;
    }

    QStatusBar {
        background-color: #2e2e2e;
        color: #ffffff;
    }
"""

class CodeExecutionWorker(QThread):
    execution_done = pyqtSignal(str, str)  # Signal to emit output and error

    def __init__(self, code, globals_dict, locals_dict):
        super().__init__()
        self.code = code
        self.globals_dict = globals_dict
        self.locals_dict = locals_dict

    def run(self):
        try:
            # First attempt eval (for expressions)
            output = str(eval(self.code, self.globals_dict, self.locals_dict))
            error = ""
        except SyntaxError:
            # If it’s not an expression, run it as a statement using exec
            try:
                exec(self.code, self.globals_dict, self.locals_dict)
                output = ""
                error = ""
            except Exception as e:
                output = ""
                error = str(e)
        except Exception as e:
            output = ""
            error = str(e)

        # Emit the result and any error message back to the main thread
        self.execution_done.emit(output, error)


class PyQtGraphCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        # Create a layout for the widget
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        # Create a PlotWidget for the image and segmentation views
        self.img_plot = pg.PlotWidget(title="Image", border="w")
        self.seg_plot = pg.PlotWidget(title="Segmentation", border="w")

        self.img_plot.getViewBox().invertY(True)
        self.seg_plot.getViewBox().invertY(True)
        
        # Add the plots to the layout
        self.layout.addWidget(self.img_plot)
        self.layout.addWidget(self.seg_plot)

        # Initialize data - 512x512 with white outline
        self.img_data = np.ones((512, 512, 3), dtype=np.uint8)
        self.img_data[5:-5, 5:-5] = 0 # white border
        self.seg_data = self.img_data.copy()


        # Plot the data
        self.img_item = pg.ImageItem(self.img_data)
        self.seg_item = pg.ImageItem(self.seg_data)
        self.img_overlay=pg.ImageItem()
        self.seg_overlay=pg.ImageItem()

        self.img_plot.addItem(self.img_item)
        self.seg_plot.addItem(self.seg_item)

        # Add the overlay to both plots
        self.img_plot.addItem(self.img_overlay)
        self.seg_plot.addItem(self.seg_overlay)

        self.img_plot.setAspectLocked(True)
        self.seg_plot.setAspectLocked(True)

        # Set initial zoom levels
        self.img_plot.setRange(xRange=[0, self.img_data.shape[1]], yRange=[0, self.img_data.shape[0]], padding=0)
        self.seg_plot.setRange(xRange=[0, self.seg_data.shape[1]], yRange=[0, self.seg_data.shape[0]], padding=0)

        # Connect the range change signals to the custom slots
        self.img_plot.sigRangeChanged.connect(self.sync_seg_plot)
        self.seg_plot.sigRangeChanged.connect(self.sync_img_plot)

        # Connect the mouse move signals to the custom slots
        self.img_plot.scene().sigMouseMoved.connect(self.update_cursor)
        self.seg_plot.scene().sigMouseMoved.connect(self.update_cursor)

        # Create crosshair lines
        self.img_vline = pg.InfiniteLine(angle=90, movable=False)
        self.img_hline = pg.InfiniteLine(angle=0, movable=False)
        self.seg_vline = pg.InfiniteLine(angle=90, movable=False)
        self.seg_hline = pg.InfiniteLine(angle=0, movable=False)

        self.img_plot.addItem(self.img_vline, ignoreBounds=True)
        self.img_plot.addItem(self.img_hline, ignoreBounds=True)
        self.seg_plot.addItem(self.seg_vline, ignoreBounds=True)
        self.seg_plot.addItem(self.seg_hline, ignoreBounds=True)

    def highlight_cells(self, cell_indices, alpha=0.3, color='white', cell_colors=None, segmentation_outline=False):
        from matplotlib import colors
        if cell_colors is None: # single color mode
            color=[*colors.to_rgb(color), alpha] # convert color to RGBA
            mask=np.isin(self.parent.frame.masks-1, cell_indices)[..., np.newaxis]*color
        else: # multi-color mode
            mask=np.zeros((*self.parent.frame.masks.shape, 4))
            cell_colors=[[*colors.to_rgb(c), alpha] for c in cell_colors]
            for i, cell_index in enumerate(cell_indices):
                mask[np.isin(self.parent.frame.masks-1,cell_index)]=cell_colors[i]

        if segmentation_outline:
            outlines=(self.parent.frame.outlines.todense()!=0)
            mask[outlines]=[1,1,1,0.5]
        
        mask=np.rot90(mask, 3)
        mask=np.fliplr(mask)

        opaque_mask=mask.copy()
        opaque_mask[mask[...,-1]!=0, -1]=1

        self.img_overlay.setImage(mask)
        self.seg_overlay.setImage(opaque_mask)

    def clear_overlay(self):
        self.img_overlay.clear()
        self.seg_overlay.clear()

    def get_plot_coords(self, pos, pixels=True):
        """Get the pixel coordinates of the mouse cursor."""
        mouse_point = self.img_plot.plotItem.vb.mapSceneToView(pos) # axes are the same for both plots so we can use either to transform
        x, y = mouse_point.x(), mouse_point.y()
        if pixels:
            x, y = int(y), int(x)
        return x, y
    
    def update_cursor(self, pos):
        """Update the segmentation plot cursor based on the image plot cursor."""
        #if self.img_plot.sceneBoundingRect().contains(pos):
        #    mouse_point = self.img_plot.plotItem.vb.mapSceneToView(pos)
        #elif self.seg_plot.sceneBoundingRect().contains(pos):
        #    mouse_point = self.seg_plot.plotItem.vb.mapSceneToView(pos)
        x,y=self.get_plot_coords(pos, pixels=False)
        self.seg_vline.setPos(x)
        self.seg_hline.setPos(y)
        self.img_vline.setPos(x)
        self.img_hline.setPos(y)
        self.parent.update_coordinate_label(int(x), int(y))

    def update_display(self, img_data=None, seg_data=None):
        """Update the display when checkboxes change."""
        if img_data is not None:
            self.img_data = np.rot90(img_data, 3)
            # invert y axis
            self.img_data = np.fliplr(self.img_data)
        if seg_data is not None:
            self.seg_data = np.rot90(seg_data, 3)
            # invert y axis
            self.seg_data = np.fliplr(self.seg_data)

        # RGB checkboxes
        RGB_checks = self.get_RGB()
        for i, check in enumerate(RGB_checks):
            if not check:
                self.img_data[..., i] = 0
            
        # Grayscale checkbox
        if self.parent.grayscale.isChecked():
            self.img_data = np.mean(self.img_data, axis=-1)

        self.img_item.setImage(self.img_data)
        self.seg_item.setImage(self.seg_data)

    def sync_img_plot(self, view_box):
        """Sync the image plot view range with the segmentation plot."""
        self.img_plot.blockSignals(True)
        self.img_plot.setRange(xRange=self.seg_plot.viewRange()[0], yRange=self.seg_plot.viewRange()[1], padding=0)
        self.img_plot.blockSignals(False)

    def sync_seg_plot(self, view_box):
        """Sync the segmentation plot view range with the image plot."""
        self.seg_plot.blockSignals(True)
        self.seg_plot.setRange(xRange=self.img_plot.viewRange()[0], yRange=self.img_plot.viewRange()[1], padding=0)
        self.seg_plot.blockSignals(False)

    def get_RGB(self):
        return [checkbox.isChecked() for checkbox in self.parent.RGB_checkboxes]

    def set_RGB(self, RGB):
        if isinstance(RGB, bool):
            RGB=[RGB]*3
        elif len(RGB)!=3:
            raise ValueError('RGB must be a bool or boolean array of length 3.')
        for checkbox, state in zip(self.parent.RGB_checkboxes, RGB):
            checkbox.setChecked(state)

class MainWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.file_loaded = False
        self.setWindowTitle("PyQtGraph Segmentation Viewer")
        self.resize(1080, 540)

        # ----------------Toolbar items----------------
        self.spacer = QSpacerItem(20, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # FUCCI
        self.FUCCI_checkbox = QCheckBox("Show FUCCI Channel", self)
        self.cc_overlay_dropdown = QComboBox(self)
        self.cc_overlay_dropdown.addItems(["None", "Green", "Red", "All"])
        self.overlay_label = QLabel("FUCCI overlay", self)

        # RGB
        self.RGB_checkbox_widget = QWidget()
        self.RGB_layout = QHBoxLayout(self.RGB_checkbox_widget)
        self.RGB_checkboxes = [QCheckBox(s, self) for s in ['R', 'G', 'B']]
        for checkbox in self.RGB_checkboxes:
            checkbox.setChecked(True)
            self.RGB_layout.addWidget(checkbox)
        self.grayscale=QCheckBox("Grayscale", self)

        # Normalize
        self.normalize_label = QLabel("Normalize by:", self)
        self.normalize_widget=QWidget()
        self.normalize_layout=QHBoxLayout(self.normalize_widget)
        self.normalize_frame_button=QRadioButton("Frame", self)
        self.normalize_stack_button=QRadioButton("Stack", self)
        self.normalize_layout.addWidget(self.normalize_frame_button)
        self.normalize_layout.addWidget(self.normalize_stack_button)
        self.normalize_frame_button.setChecked(True)
        self.normalize_stack=False

        # Command Line Interface
        self.command_line_button=QPushButton("Open Command Line", self)
        self.globals_dict = {'main': self, 'np': np, 'preprocessing': preprocessing, 'tqdm': tqdm}
        self.locals_dict = {}

        # Save Menu
        self.save_menu_widget=QWidget()
        self.save_menu=QHBoxLayout(self.save_menu_widget)
        self.save_button = QPushButton("Save", self)
        self.save_as_button = QPushButton("Save As", self)
        self.save_menu.addWidget(self.save_button)
        self.save_menu.addWidget(self.save_as_button)
        self.save_stack = QCheckBox("Save Stack", self)

        # Status bar
        self.status_coordinates=QLabel("Cursor: (x, y)", self)
        self.status_cell=QLabel("Selected Cell: None", self)
        self.status_frame_number=QLabel("Frame: None", self)
        self.statusBar().addWidget(self.status_coordinates)
        self.statusBar().addWidget(self.status_cell)
        self.statusBar().addWidget(self.status_frame_number)

        #----------------Layout----------------
        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        self.toolbar = QWidget()
        self.toolbar_layout = QVBoxLayout(self.toolbar)
        self.canvas_widget = self.get_canvas()
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas_widget)

        # Toolbar widgets
        self.toolbar_layout.setSpacing(0)
        self.toolbar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.toolbar.setFixedWidth(150)

        self.toolbar_layout.addWidget(self.overlay_label)
        self.toolbar_layout.addWidget(self.cc_overlay_dropdown)
        self.toolbar_layout.addWidget(self.FUCCI_checkbox)

        self.toolbar_layout.addItem(self.spacer)

        self.toolbar_layout.addWidget(self.RGB_checkbox_widget)
        self.toolbar_layout.addWidget(self.grayscale)

        self.toolbar_layout.addItem(self.spacer)

        self.toolbar_layout.addWidget(self.normalize_label)
        self.toolbar_layout.addWidget(self.normalize_widget)

        self.toolbar_layout.addItem(self.spacer)

        self.toolbar_layout.addWidget(self.command_line_button)

        self.toolbar_layout.addStretch() # spacer between top and bottom aligned widgets

        self.toolbar_layout.addWidget(self.save_menu_widget)
        self.toolbar_layout.addWidget(self.save_stack)

        self.file_loaded = False
        
        #----------------Connections----------------
        self.cc_overlay_dropdown.currentIndexChanged.connect(self.cc_overlay)
        self.FUCCI_checkbox.stateChanged.connect(self.update_display)
        self.save_button.clicked.connect(self.save_segmentation)
        self.save_as_button.clicked.connect(self.save_as_segmentation)
        for checkbox in self.RGB_checkboxes:
            checkbox.stateChanged.connect(self.update_display)
        self.grayscale.stateChanged.connect(self.update_display)
        self.normalize_frame_button.toggled.connect(self.update_normalize_frame)
        self.command_line_button.clicked.connect(self.open_command_line)

        # click event
        self.canvas.img_plot.scene().sigMouseClicked.connect(self.on_click)
        self.canvas.seg_plot.scene().sigMouseClicked.connect(self.on_click)
    
    def update_coordinate_label(self, x, y):
        self.status_coordinates.setText(f"Coordinates: ({x}, {y})")
    
    def update_cell_label(self, cell_n):
        if cell_n is None:
            self.status_cell.setText("Selected Cell: None")
        else:
            self.status_cell.setText(f"Selected Cell: {cell_n}")

    def update_display(self):
        """Redraw the image data with whatever new settings have been applied from the toolbar."""
        if self.file_loaded:
            img_data=self.normalize(self.frame.img)
            self.canvas.update_display(img_data=img_data, seg_data=self.frame.outlines.todense()!=0)
    
    def update_normalize_frame(self):
        self.normalize_stack=self.normalize_stack_button.isChecked()
        self.update_display()

    def normalize(self, img):
        if self.normalize_stack and self.stack is not None:
            if hasattr(self.stack, 'bounds'):
                bounds=self.stack.bounds
            else:
                all_imgs=np.array([frame.img for frame in self.stack.frames]).reshape(-1, 3)
                bounds=np.quantile(all_imgs, (0.01, 0.99), axis=0).T
                self.stack.bounds=bounds

            normed_img=preprocessing.normalize_RGB(img, bounds=bounds)
        else:
            if hasattr(self.frame, 'bounds'):
                bounds=self.frame.bounds
            else:
                bounds=np.quantile(img.reshape(-1,3), (0.01, 0.99), axis=0).T
                self.frame.bounds=bounds
            normed_img=preprocessing.normalize(img, bounds=bounds)

        return normed_img
    
    def open_command_line(self):
        # Create a separate window for the command line interface
        self.cli_window = QMainWindow()
        self.cli_window.setWindowTitle("Command Line Interface")

        # Add the CommandLineWidget to the new window
        self.cli_widget = CommandLineWidget(parent=self, globals_dict=self.globals_dict, locals_dict=self.locals_dict)
        self.globals_dict['cli'] = self.cli_widget
        self.cli_window.setCentralWidget(self.cli_widget)

        # Set the window size and show the window
        self.cli_window.resize(700, 400)
        self.cli_window.show()

    def on_click(self, event):
        if not self.file_loaded:
            return
        
        x, y = self.canvas.get_plot_coords(event.scenePos(), pixels=True)
        current_cell_n = self.get_cell(x, y)
        if current_cell_n>=0:
            overlay_color=self.cc_overlay_dropdown.currentText().lower()
            if overlay_color=='none': # simple highlighting mode
                    if current_cell_n==self.selected_cell:
                        self.selected_cell=None
                    else:
                        self.selected_cell=current_cell_n
                    self.canvas.highlight_cells([self.selected_cell], alpha=0.3, color='white')
            else:
                self.selected_cell=current_cell_n
                cell=self.frame.cells[current_cell_n]
                if event.button() == Qt.MouseButton.LeftButton:
                    cell.green=not cell.green
                if event.button() == Qt.MouseButton.RightButton:
                    cell.red=not cell.red
                if event.button() == Qt.MouseButton.MiddleButton:
                    if cell.red and cell.green: # if orange, reset to none
                        cell.green=False
                        cell.red=False
                    else: # otherwise, set to orange
                        cell.green=True
                        cell.red=True
                self.cc_overlay()
        else:
            self.selected_cell=None # background
            self.canvas.highlight_cells([self.selected_cell], alpha=0.3, color='white')
        
        if hasattr(self.stack, 'tracked_centroids'):
            t=self.stack.tracked_centroids
            self.selected_particle=t[(t.frame==self.frame.frame_number)&(t.cell_number==self.selected_cell)]['particle']
            if len(self.selected_particle)==0:
                self.selected_particle=None
            elif len(self.selected_particle)==1:
                self.selected_particle=self.selected_particle.item()
            else:
                raise ValueError(f'Multiple particles found for cell {self.selected_cell} in frame {self.frame.frame_number}')

        self.update_cell_label(self.selected_cell)

    
    def get_cell(self, x, y):
        if x < 0 or y < 0 or x >= self.canvas.img_data.shape[1] or y >= self.canvas.img_data.shape[0]:
            return -1 # out of bounds
        cell_n=self.frame.masks[x, y]-1
        return cell_n
        
    def save_segmentation(self, stack=False):
        if not self.file_loaded:
            return
        
        if stack:
            frames_to_save=self.stack.frames
        else:
            frames_to_save=[self.stack.frames[self.frame_number]]

        for frame in frames_to_save:
            try:
                green, red=np.array(frame.get_cell_attr(['green', 'red'])).T
                frame.cell_cycles=green+2*red
                frame.to_seg_npy(write_attrs=['cell_cycles'])
            except AttributeError:
                frame.to_seg_npy()
            print(f'Saved frame to {frame.name}')


    def save_as_segmentation(self):
        if not self.file_loaded:
            return

        frame=self.stack.frames[self.frame_number]
        file_path=QFileDialog.getSaveFileName(self, 'Save segmentation as...', filter='*_seg.npy')[0]
        frame.to_seg_npy(file_path)
        print(f'Saved segmentation to {file_path}')

    def get_canvas(self):
        """Initialize the PyQtGraph canvas."""
        self.canvas = PyQtGraphCanvas(parent=self)
        canvas_widget = QWidget()
        self.canvas_layout = QVBoxLayout(canvas_widget)
        self.canvas_layout.addWidget(self.canvas)
        
        return canvas_widget

    def cc_overlay(self):
        """Handle cell cycle overlay options."""
        if not self.file_loaded:
            return
        
        overlay_color=self.cc_overlay_dropdown.currentText().lower()
        if overlay_color == 'none': # clear overlay
            for plot_widget in [self.canvas.img_plot, self.canvas.seg_plot]:
                plot_widget.setMenuEnabled(True) # re-enable menus
            self.selected_cell=None
            self.update_cell_label(None)
            self.canvas.highlight_cells([])
        else:
            for plot_widget in [self.canvas.img_plot, self.canvas.seg_plot]:
                plot_widget.setMenuEnabled(False) # disable menus (custom right-click events)
            if overlay_color == 'all':
                colors=np.array(['g','r','orange'])
                green, red=np.array(self.frame.get_cell_attr(['green', 'red'])).T
                colored_cells=np.where(red | green)[0] # cells that are either red or green
                cell_cycle=green+2*red-1
                cell_colors=colors[cell_cycle[colored_cells]] # map cell cycle state to green, red, orange
                self.canvas.highlight_cells(colored_cells, alpha=0.1, cell_colors=cell_colors, segmentation_outline=True)

            else:
                colored_cells=np.where(self.frame.get_cell_attr(overlay_color))[0]
                self.canvas.highlight_cells(colored_cells, alpha=0.1, color=overlay_color, segmentation_outline=True)

    # Drag and drop event
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def reset_display(self):
        self.selected_cell=None
        self.update_cell_label(None)
        self.cc_overlay_dropdown.setCurrentIndex(0) # clear overlay
        self.canvas.set_RGB(True)
        self.grayscale.setChecked(False)
        self.canvas.clear_overlay() # remove any overlays (highlighting, outlines)
        self.canvas.img_plot.autoRange()

    def imshow(self, frame, reset=True):
        ''' Display the image and segmentation data for a given frame. Should be run once per loading a new frame.'''
        # reset toolbar
        self.status_frame_number.setText(f'Frame: {frame.frame_number}')
        self.frame = frame
        self.globals_dict['frame']=frame
        if reset: 
            self.reset_display()
        else:
            if hasattr(self, 'selected_particle'):
                # TODO: non-redundant workflow for indexing cells and particles
                t=self.stack.tracked_centroids
                self.selected_cell=t[(t.frame==self.frame.frame_number)&(t.particle==self.selected_particle)]['cell_number']
                if len(self.selected_cell)==0:
                    self.selected_cell=None
                elif len(self.selected_cell)==1:
                    self.selected_cell=self.selected_cell.item()
                else:
                    raise ValueError(f'Multiple cells found for particle {self.particle} in frame {self.frame.frame_number}')
                
                self.update_cell_label(self.selected_cell)
                self.canvas.highlight_cells([self.selected_cell], alpha=0.3, color='white')
                
            else:
                self.canvas.clear_overlay() # no tracking data, clear highlights

        if not hasattr(self.frame.cells[0], 'green'):
            self.get_red_green()
        if self.cc_overlay_dropdown.currentIndex() != 0:
            self.cc_overlay()

        self.update_display()
    
    def update_RGB(self):
        """Update the display when the FUCCI checkbox changes."""
        self.canvas.update_display()

    def dropEvent(self, event):
        from natsort import natsorted
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        self.load_files(natsorted(files))
        if self.file_loaded:
            self.frame_number = 0
            self.imshow(self.stack.frames[self.frame_number])
            self.canvas.img_plot.autoRange()
            self.globals_dict['stack']=self.stack
        

    def keyPressEvent(self, event):
        """Handle key press events (e.g., arrow keys for frame navigation)."""
        if not self.file_loaded:
            return

        # Ctrl-key shortcuts
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # FUCCI labeling modes
            if event.key() == Qt.Key.Key_R:
                if self.cc_overlay_dropdown.currentIndex() == 2:
                    self.cc_overlay_dropdown.setCurrentIndex(0)
                    self.canvas.set_RGB(True)
                else:
                    self.cc_overlay_dropdown.setCurrentIndex(2)
                    self.canvas.set_RGB([True, False, False])
                return
            
            elif event.key() == Qt.Key.Key_G:
                if self.cc_overlay_dropdown.currentIndex() == 1:
                    self.cc_overlay_dropdown.setCurrentIndex(0)
                    self.canvas.set_RGB(True)
                else:
                    self.cc_overlay_dropdown.setCurrentIndex(1)
                    self.canvas.set_RGB([False, True, False])
                return
            
            elif event.key() == Qt.Key.Key_B:
                if np.array_equal(self.canvas.get_RGB(), [False, False, True]):
                    self.canvas.set_RGB(True)
                else:
                    self.canvas.set_RGB([False, False, True])
                    self.cc_overlay_dropdown.setCurrentIndex(0)
                return
            
            elif event.key() == Qt.Key.Key_A:
                if self.cc_overlay_dropdown.currentIndex() == 3:
                    self.cc_overlay_dropdown.setCurrentIndex(0)
                    self.canvas.set_RGB(True)
                else:
                    self.cc_overlay_dropdown.setCurrentIndex(3)
                    self.canvas.set_RGB([True, True, False])
                return

            # save (with overwrite)
            elif event.key() == Qt.Key.Key_S:
                self.save_segmentation(stack=self.save_stack.isChecked())
                return
        
        # ctrl+shift shortcuts
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier:
            if event.key() == Qt.Key.Key_S:
                self.save_as_segmentation()
                return

        # r-g-b toggles
        if event.key() == Qt.Key.Key_R:
            self.RGB_checkboxes[0].toggle()
        elif event.key() == Qt.Key.Key_G:
            self.RGB_checkboxes[1].toggle()
        elif event.key() == Qt.Key.Key_B:
            self.RGB_checkboxes[2].toggle()
        
        elif event.key() == Qt.Key.Key_Tab:
            # toggle grayscale
            self.grayscale.toggle()

        # reset visuals
        if event.key() == Qt.Key.Key_Escape:
            self.cc_overlay_dropdown.setCurrentIndex(0)
            self.canvas.set_RGB(True)
            self.canvas.img_plot.autoRange()
            self.grayscale.setChecked(False)

        # Handle frame navigation with left and right arrow keys
        if event.key() == Qt.Key.Key_Left:
            if self.frame_number > 0:
                self.frame_number -= 1
                self.imshow(self.stack.frames[self.frame_number], reset=False)

        elif event.key() == Qt.Key.Key_Right:
            if self.frame_number < len(self.stack.frames) - 1:
                self.frame_number += 1
                self.imshow(self.stack.frames[self.frame_number], reset=False)

    def load_files(self, files):
        """Load segmentation files."""
        self.file_loaded = True
        if files[0].endswith('seg.npy'): # segmented file paths
            if len(files)==1: # single file
                progress_bar=lambda x: x
            else:
                progress_bar=tqdm
            self.stack=TimeSeries(frame_paths=files, load_img=True, progress_bar=progress_bar)
        else:
            try: # maybe it's a folder of segmented files?
                self.stack=TimeSeries(stack_path=files[0], verbose_load=True, progress_bar=tqdm, load_img=True)
            except FileNotFoundError: # nope, just reject it
                print(f'ERROR: File {files[0]} is not a seg.npy file, cannot be loaded.')
                self.file_loaded = False

    def get_red_green(self):
        ''' Fetch or create red and green attributes for cells in the current frame. '''
        for cell in self.frame.cells:
            if hasattr(cell, 'cycle_stage'):
                cell.green=cell.cycle_stage==1 or cell.cycle_stage==3
                cell.red=cell.cycle_stage==2 or cell.cycle_stage==3
            else:
                cell.red=False
                cell.green=False

    def closeEvent(self, event):
        # Close the command line window when the main window is closed
        if hasattr(self, 'cli_window'):
            self.cli_window.close()
        event.accept()

class CommandLineWidget(QWidget):
    def __init__(self, parent=None, globals_dict={}, locals_dict={}):
        super().__init__(parent)

        # Set up the layout
        self.layout = QVBoxLayout(self)
        
        # Terminal-style output display area (Read-Only)
        self.terminal_display = QTextEdit(self)
        self.terminal_display.setStyleSheet("""
            background-color: black;
            color: white;
            font-family: "Courier";
            font-size: 10pt;
        """)
        self.terminal_display.setReadOnly(True)
        self.layout.addWidget(self.terminal_display)
        
        # Command input area
        self.command_input = QLineEdit(self)
        self.command_input.setStyleSheet("""
            background-color: black;
            color: white;
            font-family: "Courier";
            font-size: 12pt;
        """)
        #self.command_input.setPlaceholderText("Type your command here and press Enter")
        self.layout.addWidget(self.command_input)

        # Command history
        self.command_history = []
        self.history_index = -1

        # Connect Enter key press to command execution
        self.command_input.returnPressed.connect(self.execute_command)

        self.globals_dict = globals_dict
        self.locals_dict = locals_dict

        # Prompt for commands
        self.prompt = ">>> "

    def showEvent(self, event):
        super().showEvent(event)
        # Set the focus to the command input box when the window is shown
        self.command_input.setFocus()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:
            # Navigate through command history (up)
            if self.history_index > 0:
                self.history_index -= 1
                self.command_input.setText(self.command_history[self.history_index])
        elif event.key() == Qt.Key_Down:
            # Navigate through command history (down)
            if self.history_index < len(self.command_history) - 1:
                self.history_index += 1
                self.command_input.setText(self.command_history[self.history_index])
            else:
                self.history_index = len(self.command_history)  # Reset to allow new input
                self.command_input.clear()
        else:
            # Call base class keyPressEvent for default handling
            super().keyPressEvent(event)
    
    def execute_command(self):
        # Get the command from the input box
        command = self.command_input.text()
        self.command_input.clear()
        
        if command:
            self.command_history.append(command)
            self.history_index = len(self.command_history)  # Reset index to point to the latest command
            # Display the command in the terminal display
            self.terminal_display.append(self.prompt + command)
            
            # Execute the command and show the result
            self.worker = CodeExecutionWorker(command, self.globals_dict, self.locals_dict)
            self.worker.execution_done.connect(self.on_code_execution_done)
            self.worker.start()

    @pyqtSlot(str, str) # Decorator to specify the type of the signal
    def on_code_execution_done(self, output, error):
        if output:
            self.terminal_display.append(output)
        if error:
            self.terminal_display.append(f"Error: {error}")
            
if not QApplication.instance():
    app = QApplication(sys.argv)
else:
    app = QApplication.instance()
app.setStyleSheet(dark_mode_style)
app.quitOnLastWindowClosed = True
ui = MainWidget()
ui.show()
app.exec()