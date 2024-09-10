import sys
import numpy as np
from tqdm import tqdm

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QComboBox, QPushButton,
    QVBoxLayout, QHBoxLayout, QCheckBox, QSpacerItem, QSizePolicy
)
from PyQt6.QtGui import QIcon, QMouseEvent
from PyQt6.QtCore import Qt

from vispy.scene import PanZoomCamera
from vispy import scene
from vispy.visuals.transforms import STTransform

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseButton

from monolayer_tracking.segmented_comprehension import Image, TimeSeries
from monolayer_tracking import preprocessing


class VispyCanvas(scene.SceneCanvas):
    def __init__(self, parent=None, width=800, height=600):
        # Initialize the SceneCanvas
        super().__init__(keys='interactive', size=(width, height), show=True)

        # Unfreeze the canvas so that attributes can be added
        self.unfreeze()
        self.parent_widget=parent
        # Create a grid layout for two views (image and segmentation)
        grid = self.central_widget.add_grid(spacing=10)

        # Create two viewboxes for image and segmentation
        self.img_view = grid.add_view(row=0, col=0, border_color='white')
        self.img_view.camera = PanZoomCamera(aspect=1)
        self.img_view.camera.interactive = False

        self.seg_view = grid.add_view(row=0, col=1, border_color='white')
        self.seg_view.camera = self.img_view.camera  # Link the cameras for synchronized pan/zoom

        # Add visuals for image and segmentation
        self.img_data = np.random.randint(0, 256, (500, 500))
        self.seg_data = (np.random.random((500, 500)) > 128)*255

        self.img_visual = scene.visuals.Image(self.img_data, parent=self.img_view.scene, cmap='gray')
        self.seg_visual = scene.visuals.Image(self.img_data, parent=self.seg_view.scene, cmap='gray')

        # Freeze the canvas again after adding the attributes
        self.freeze()

    def update_image(self, new_img_data, new_seg_data):
        """Update the displayed image and segmentation."""
        self.img_visual.set_data(new_img_data)
        self.seg_visual.set_data(new_seg_data)
        self.img_view.camera.set_range(new_img_data.shape, [-0.5, -0.5])
        self.seg_view.camera.set_range(new_seg_data.shape, [-0.5, -0.5])

    def zoom(self, scale_factor):
        """Zoom functionality."""
        for view in [self.img_view, self.seg_view]:
            view.camera.zoom(scale_factor)

    def pan(self, dx, dy):
        """Pan functionality."""
        for view in [self.img_view, self.seg_view]:
            view.camera.pan((dx, dy))

    def is_within_view(self, view, screen_pos):
        """Check if screen_pos is within the bounds of the view's visual."""
        view_bounds = view.scene.bounding_box
        view_pos = view.camera.screen_transform.map(screen_pos)
        x, y = view_pos[:2]
        return (view_bounds[0] <= x <= view_bounds[2]) and (view_bounds[1] <= y <= view_bounds[3])


class MainWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.file_loaded = False
        self.setWindowTitle("VisPy Segmentation Viewer")
        self.resize(720, 480)
        self.is_panning = False

        # Menu items (using PyQt)
        self.menu_FUCCI_checkbox = QCheckBox("Show FUCCI Channel", self)
        self.menu_cc_overlay_dropdown = QComboBox(self)
        self.menu_cc_overlay_dropdown.addItems(["None", "Green", "Red", "All"])
        self.menu_overlay_label = QLabel("FUCCI overlay", self)
        self.spacer = QSpacerItem(20, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.save_button = QPushButton("Save", self)

        self.menu_RGB_checkbox_widget = QWidget()
        self.menu_RGB_layout = QHBoxLayout(self.menu_RGB_checkbox_widget)
        self.menu_RGB_checkboxes = [QCheckBox(s, self) for s in ['R', 'G', 'B']]
        for checkbox in self.menu_RGB_checkboxes:
            checkbox.setChecked(True)
            self.menu_RGB_layout.addWidget(checkbox)

        # Link menu items to methods
        self.menu_cc_overlay_dropdown.currentIndexChanged.connect(self.cc_overlay)
        self.menu_FUCCI_checkbox.stateChanged.connect(self.update_display)
        self.save_button.clicked.connect(self.save_segmentation)
        for checkbox in self.menu_RGB_checkboxes:
            checkbox.stateChanged.connect(self.update_display)

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
        self.toolbar_layout.addWidget(self.save_button)
        self.toolbar_layout.addWidget(self.menu_RGB_checkbox_widget)

        self.file_loaded = False

    def save_segmentation(self):
        if not self.file_loaded:
            return
        # This is a placeholder for saving the segmentation data
        print("Segmentation saved.")

    def get_canvas(self):
        """Initialize the VisPy canvas."""
        self.canvas = VispyCanvas()
        canvas_widget = QWidget()
        self.canvas_layout = QVBoxLayout(canvas_widget)
        self.canvas_layout.addWidget(self.canvas.native)
        
        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.mouse_release.connect(self.on_mouse_release)
        self.canvas.events.mouse_move.connect(self.on_mouse_move)
        self.canvas.events.mouse_wheel.connect(self.on_mouse_wheel)
        return canvas_widget

    def update_display(self):
        """Update the display when checkboxes change."""
        if self.file_loaded:
            # Placeholder: Update the visuals on checkbox change
            print("Display updated.")

    def cc_overlay(self):
        """Handle cell cycle overlay options."""
        # Placeholder: Handle changes in the dropdown for overlays
        print("Cell cycle overlay changed.")

    def mouse_press_img(self, event):
        print("Image mouse press")
    
    def mouse_press_seg(self, event):
        print("Segmentation mouse press")

    # Drag and drop event
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        self.load_files(files)
        self.frame_number = 0

    def keyPressEvent(self, event):
        """Handle key press events (e.g., arrow keys for frame navigation)."""
        if not self.file_loaded:
            return
        # Placeholder: Handle frame navigation

    def load_files(self, files):
        """Load segmentation files."""
        # Placeholder: Add file loading logic
        self.file_loaded = True
        print(f"Loaded files: {files}")

    def pan(self, dx, dy):
        """Pan the view when right-click dragging."""
        if self.is_panning:
            self.canvas.pan(dx, -dy)

    def on_mouse_press(self, event):
        """Handle mouse press events."""
        if event.button==1:
            print(event.pos)

        if event.button==2:
            self.is_panning = True
            self.pan_start_pos = event.pos

    def on_mouse_release(self, event):
        """Handle mouse release events."""
        if event.button==2:
            self.is_panning = False

    def on_mouse_move(self, event):
        """Handle mouse move events."""
        if self.is_panning:
            dx = event.pos[0] - self.pan_start_pos[0]
            dy = event.pos[1] - self.pan_start_pos[1]
            self.pan(-dx, -dy)
            self.pan_start_pos = event.pos

    def on_mouse_wheel(self, event):
        """Handle mouse wheel events."""
        scale_factor=1.2
        #if event.delta[1] > 0:  # Scroll up
        self.canvas.zoom(scale_factor**(-event.delta[1])) # Zoom in
        #else:  # Scroll down
        #    self.canvas.zoom(scale_factor)  # Zoom out

if __name__ == '__main__':
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    app.quitOnLastWindowClosed = True
    ui = MainWidget()
    ui.show()
    sys.exit(app.exec())