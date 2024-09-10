import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QComboBox, QPushButton,
    QVBoxLayout, QHBoxLayout, QCheckBox, QSpacerItem, QSizePolicy, QFileDialog
)
from PyQt6.QtCore import Qt
import pyqtgraph as pg

from monolayer_tracking.segmented_comprehension import Image, TimeSeries
from monolayer_tracking import preprocessing

from tqdm import tqdm

class PyQtGraphCanvas(QWidget):
    def __init__(self, parent=None, width=800, height=600):
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
            mask[outlines]=[1,1,1,alpha]
        
        mask=np.rot90(mask, 3)
        mask=np.fliplr(mask)

        opaque_mask=mask.copy()
        opaque_mask[mask[...,-1]!=0, -1]=1

        self.img_overlay.setImage(mask)
        self.seg_overlay.setImage(opaque_mask)

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
        if self.parent.menu_grayscale.isChecked():
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
        return [checkbox.isChecked() for checkbox in self.parent.menu_RGB_checkboxes]

    def set_RGB(self, RGB):
        if isinstance(RGB, bool):
            RGB=[RGB]*3
        elif len(RGB)!=3:
            raise ValueError('RGB must be a bool or boolean array of length 3.')
        for checkbox, state in zip(self.parent.menu_RGB_checkboxes, RGB):
            checkbox.setChecked(state)

class MainWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.file_loaded = False
        self.setWindowTitle("PyQtGraph Segmentation Viewer")
        self.resize(720, 480)

        # Toolbar items (using PyQt)
        self.menu_FUCCI_checkbox = QCheckBox("Show FUCCI Channel", self)
        self.menu_cc_overlay_dropdown = QComboBox(self)
        self.menu_cc_overlay_dropdown.addItems(["None", "Green", "Red", "All"])
        self.menu_overlay_label = QLabel("FUCCI overlay", self)
        self.spacer = QSpacerItem(20, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.save_menu_widget=QWidget()
        self.save_menu=QHBoxLayout(self.save_menu_widget)
        self.save_button = QPushButton("Save", self)
        self.save_as_button = QPushButton("Save As", self)
        self.save_menu.addWidget(self.save_button)
        self.save_menu.addWidget(self.save_as_button)
        self.menu_RGB_checkbox_widget = QWidget()
        self.menu_RGB_layout = QHBoxLayout(self.menu_RGB_checkbox_widget)
        self.menu_RGB_checkboxes = [QCheckBox(s, self) for s in ['R', 'G', 'B']]
        for checkbox in self.menu_RGB_checkboxes:
            checkbox.setChecked(True)
            self.menu_RGB_layout.addWidget(checkbox)
        self.menu_grayscale=QCheckBox("Grayscale", self)

        self.status_coordinates=QLabel("Cursor: (x, y)", self)
        self.status_cell=QLabel("Selected Cell: None", self)
        self.statusBar().addWidget(self.status_coordinates)
        self.statusBar().addWidget(self.status_cell)

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
        self.toolbar_layout.addWidget(self.menu_FUCCI_checkbox)
        self.toolbar_layout.addItem(self.spacer)
        self.toolbar_layout.addWidget(self.menu_overlay_label)
        self.toolbar_layout.addWidget(self.menu_cc_overlay_dropdown)
        self.toolbar_layout.addItem(self.spacer)
        self.toolbar_layout.addWidget(self.menu_RGB_checkbox_widget)
        self.toolbar_layout.addWidget(self.menu_grayscale)
        self.toolbar_layout.addWidget(self.save_menu_widget)

        self.file_loaded = False
        
        # Link menu items to methods
        self.menu_cc_overlay_dropdown.currentIndexChanged.connect(self.cc_overlay)
        self.menu_FUCCI_checkbox.stateChanged.connect(self.update_display)
        self.save_button.clicked.connect(self.save_segmentation)
        self.save_as_button.clicked.connect(self.save_as_segmentation)
        for checkbox in self.menu_RGB_checkboxes:
            checkbox.stateChanged.connect(self.update_display)
        self.menu_grayscale.stateChanged.connect(self.update_display)

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
            img_data=preprocessing.normalize(self.frame.img, quantile=(0.01,0.99))
            self.canvas.update_display(img_data=img_data, seg_data=self.frame.outlines.todense()!=0)
        
    def on_click(self, event):
        if not self.file_loaded:
            return
        
        x, y = self.canvas.get_plot_coords(event.scenePos(), pixels=True)
        current_cell_n = self.get_cell(x, y)
        if current_cell_n>=0:
            overlay_color=self.menu_cc_overlay_dropdown.currentText().lower()
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
        
        self.update_cell_label(self.selected_cell)

    
    def get_cell(self, x, y):
        if x < 0 or y < 0 or x >= self.canvas.img_data.shape[1] or y >= self.canvas.img_data.shape[0]:
            return -1 # out of bounds
        cell_n=self.frame.masks[x, y]-1
        return cell_n
        
    def save_segmentation(self):
        if not self.file_loaded:
            return
        for frame in self.stack.frames:
            try:
                green, red=np.array(frame.get_cell_attr(['green', 'red'])).T
                frame.cell_cycles=green+2*red
                frame.to_seg_npy(write_attrs=['cell_cycles'])
            except AttributeError:
                frame.to_seg_npy()
            print(f'Saved segmentation to {frame.name}')

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
        
        overlay_color=self.menu_cc_overlay_dropdown.currentText().lower()
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

    def imshow(self, frame):
        ''' Display the image and segmentation data for a given frame. Should be run once per loading a new frame.'''
        # reset toolbar
        self.frame = frame
        self.selected_cell=None
        self.update_cell_label(None)
        self.menu_cc_overlay_dropdown.setCurrentIndex(0) # clear overlay
        self.canvas.set_RGB(True)
        self.menu_grayscale.setChecked(False)
        
        self.get_red_green()
        self.canvas.update_display(preprocessing.normalize(frame.img), frame.outlines.todense()!=0)
    
    def update_RGB(self):
        """Update the display when the FUCCI checkbox changes."""
        self.canvas.update_display()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        self.load_files(files)
        self.frame_number = 0
        if self.stack is not None:
            self.imshow(self.stack.frames[self.frame_number])

    def keyPressEvent(self, event):
        """Handle key press events (e.g., arrow keys for frame navigation)."""
        if not self.file_loaded:
            return

        # Ctrl-key shortcuts
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            # FUCCI labeling modes
            if event.key() == Qt.Key.Key_R:
                if self.menu_cc_overlay_dropdown.currentIndex() == 2:
                    self.menu_cc_overlay_dropdown.setCurrentIndex(0)
                    self.canvas.set_RGB(True)
                else:
                    self.menu_cc_overlay_dropdown.setCurrentIndex(2)
                    self.canvas.set_RGB([True, False, False])
                return
            
            if event.key() == Qt.Key.Key_G:
                if self.menu_cc_overlay_dropdown.currentIndex() == 1:
                    self.menu_cc_overlay_dropdown.setCurrentIndex(0)
                    self.canvas.set_RGB(True)
                else:
                    self.menu_cc_overlay_dropdown.setCurrentIndex(1)
                    self.canvas.set_RGB([False, True, False])
                return
            
            if event.key() == Qt.Key.Key_B:
                if np.array_equal(self.canvas.get_RGB(), [False, False, True]):
                    self.canvas.set_RGB(True)
                else:
                    self.canvas.set_RGB([False, False, True])
                    self.menu_cc_overlay_dropdown.setCurrentIndex(0)
                return
            
            if event.key() == Qt.Key.Key_A:
                if self.menu_cc_overlay_dropdown.currentIndex() == 3:
                    self.menu_cc_overlay_dropdown.setCurrentIndex(0)
                    self.canvas.set_RGB(True)
                else:
                    self.menu_cc_overlay_dropdown.setCurrentIndex(3)
                    self.canvas.set_RGB([True, True, False])
                return

            # save (with overwrite)
            if event.key() == Qt.Key.Key_S:
                self.save_segmentation()
                return
        
        # ctrl+shift shortcuts
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier:
            if event.key() == Qt.Key.Key_S:
                self.save_as_segmentation()
                return

        # r-g-b toggles
        if event.key() == Qt.Key.Key_R:
            self.menu_RGB_checkboxes[0].toggle()
        if event.key() == Qt.Key.Key_G:
            self.menu_RGB_checkboxes[1].toggle()
        if event.key() == Qt.Key.Key_B:
            self.menu_RGB_checkboxes[2].toggle()
        
        # reset visuals
        if event.key() == Qt.Key.Key_0:
            self.menu_cc_overlay_dropdown.setCurrentIndex(0)
            self.canvas.set_RGB(True)
            self.menu_grayscale.setChecked(False)

    def load_files(self, files):
        """Load segmentation files."""
        self.file_loaded = True
        if files[0].endswith('seg.npy'): # single segmented file
            self.stack=TimeSeries(frame_paths=files, load_img=True)
            print(f'Loaded segmentation file {files[0]}')
        else:
            try: # maybe it's a folder of segmented files?
                self.stack=TimeSeries(stack_path=files[0], verbose_load=True, progress_bar=tqdm, load_img=True)
                print(f'Loaded stack {files[0]}')
            except FileNotFoundError: # nope, just reject it
                print(f'ERROR: File {files[0]} is not a seg.npy file, cannot be loaded.')
                self.file_loaded = False

    def get_red_green(self):
        for cell in self.frame.cells:
            if hasattr(cell, 'cycle_stage'):
                cell.green=cell.cycle_stage==1 or cell.cycle_stage==3
                cell.red=cell.cycle_stage==2 or cell.cycle_stage==3
            else:
                cell.red=False
                cell.green=False

if not QApplication.instance():
    app = QApplication(sys.argv)
else:
    app = QApplication.instance()
app.quitOnLastWindowClosed = True
ui = MainWidget()
ui.show()
app.exec()