import sys
import numpy as np
import io
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QComboBox, QPushButton, QRadioButton,
    QVBoxLayout, QHBoxLayout, QCheckBox, QSpacerItem, QSizePolicy, QFileDialog,
    QLineEdit, QTextEdit, QTabWidget, QSlider, QGraphicsPolygonItem
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QPointF
from PyQt6.QtGui import QIntValidator, QPolygonF, QPen, QColor, QBrush
import pyqtgraph as pg

from monolayer_tracking.segmented_comprehension import Image, TimeSeries, Cell
from monolayer_tracking import preprocessing

from shapely.geometry import Polygon, Point

from tqdm import tqdm
# TODO: playback speed
# Later TODO: merge masks

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
    QTabWidget::pane {
        border: 1px solid #4b4b4b;
    }
    QTabBar::tab {
        background-color: #3c3c3c;  /* Unselected tab - darker shade */
        color: #ffffff;
        padding: 5px 10px;
        min-height: 20px;
        border: 1px solid #4b4b4b;
    }
    QTabBar::tab:selected {
        background-color: #5b5b5b;  /* Selected tab - medium shade */
        border-bottom: 2px solid #2b2b2b;
    }
    QTabBar::tab:hover {
        background-color: #6d6d6d;  /* Hovered tab - lighter gray */
    }
"""
class PyQtGraphCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
        # Create a layout for the widget
        plot_layout = QHBoxLayout(self)
        plot_layout.setSpacing(5)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        # Create a PlotWidget for the image and segmentation views
        self.img_plot = pg.PlotWidget(title="Image", border="w")
        self.seg_plot = pg.PlotWidget(title="Segmentation", border="w")

        self.img_plot.getViewBox().invertY(True)
        self.seg_plot.getViewBox().invertY(True)
        for plot_widget in [self.img_plot, self.seg_plot]:
            plot_widget.setMenuEnabled(False)
        # Add the plots to the layout
        plot_layout.addWidget(self.img_plot)
        plot_layout.addWidget(self.seg_plot)

        # Reduce margins around the plots
        self.img_plot.setContentsMargins(0, 0, 0, 0)
        self.seg_plot.setContentsMargins(0, 0, 0, 0)

        # Initialize data - 512x512 with white outline
        self.img_data = np.ones((512, 512, 3), dtype=np.uint8)
        self.img_data[5:-5, 5:-5] = 0 # white border
        self.seg_data = self.img_data.copy()


        # Plot the data
        self.img = pg.ImageItem(self.img_data)
        self.seg = pg.ImageItem(self.seg_data)
        self.img_outline_overlay=pg.ImageItem()
        self.mask_overlay=[pg.ImageItem(), pg.ImageItem()]
        self.selection_overlay=[pg.ImageItem(), pg.ImageItem()]

        self.img_plot.addItem(self.img)
        self.seg_plot.addItem(self.seg)

        # Add overlays to plots
        self.img_plot.addItem(self.img_outline_overlay)

        self.img_plot.addItem(self.mask_overlay[0])
        self.seg_plot.addItem(self.mask_overlay[1])

        self.img_plot.addItem(self.selection_overlay[0])
        self.seg_plot.addItem(self.selection_overlay[1])


        self.img_plot.setAspectLocked(True)
        self.seg_plot.setAspectLocked(True)

        # Connect the range change signals to the custom slots
        self.img_plot.sigRangeChanged.connect(self.sync_seg_plot)
        self.seg_plot.sigRangeChanged.connect(self.sync_img_plot)

        # Connect the mouse move signals to the custom slots
        self.img_plot.scene().sigMouseMoved.connect(self.mouse_moved)
        self.seg_plot.scene().sigMouseMoved.connect(self.mouse_moved)

        # Create crosshair lines
        self.img_vline = pg.InfiniteLine(angle=90, movable=False)
        self.img_hline = pg.InfiniteLine(angle=0, movable=False)
        self.seg_vline = pg.InfiniteLine(angle=90, movable=False)
        self.seg_hline = pg.InfiniteLine(angle=0, movable=False)
        self.img_plot.addItem(self.img_vline, ignoreBounds=True)
        self.img_plot.addItem(self.img_hline, ignoreBounds=True)
        self.seg_plot.addItem(self.seg_vline, ignoreBounds=True)
        self.seg_plot.addItem(self.seg_hline, ignoreBounds=True)

        # Set initial zoom levels
        self.img_plot.setRange(xRange=[0, self.img_data.shape[1]], yRange=[0, self.img_data.shape[0]], padding=0)
        self.seg_plot.setRange(xRange=[0, self.seg_data.shape[1]], yRange=[0, self.seg_data.shape[0]], padding=0)

    def overlay_outlines(self, event=None, color='white', alpha=0.5):
        if not self.parent.outlines_checkbox.isChecked():
            self.img_outline_overlay.clear()
            return
        from matplotlib.colors import to_rgb
        color=[*to_rgb(color), alpha]

        overlay=np.zeros((*self.parent.frame.masks.shape, 4))
        overlay[self.parent.frame.outlines]=color

        overlay=np.rot90(overlay, 3)
        overlay=np.fliplr(overlay)
        self.img_outline_overlay.setImage(overlay)

    def overlay_masks(self, event=None, cmap='tab10', alpha=0.5):
        if not self.parent.masks_checkbox.isChecked():
            # masks are not visible, clear the overlay
            self.mask_overlay[0].clear()
            self.mask_overlay[1].clear()
            return
        
        try:
            cell_colors=self.parent.frame.get_cell_attr('color_ID')
        except AttributeError:
            from matplotlib import colormaps
            #from monolayer_tracking.networks import color_masks, greedy_color
            cmap=colormaps.get_cmap(cmap)
            #random_colors=color_masks(self.parent.frame.masks)
            random_colors=np.random.randint(0, 10, size=self.parent.frame.masks.max())
            cell_colors=cmap(random_colors)[..., :3]
            self.parent.frame.set_cell_attr('color_ID', cell_colors)

        cell_indices=np.unique(self.parent.frame.masks)[1:]-1
        img_masks, seg_masks=self.highlight_cells(cell_indices, alpha=alpha, cell_colors=cell_colors, layer='mask')
        self.parent.frame.mask_overlay=[img_masks, seg_masks]

    def highlight_cells(self, cell_indices, layer='selection', alpha=0.3, color='white', cell_colors=None):
        from matplotlib.colors import to_rgb
        masks=self.parent.frame.masks

        layer=getattr(self, f'{layer}_overlay') # get the specified overlay layer: selection for highlighting, mask for colored masks

        if cell_colors is None: # single color mode
            color=[*to_rgb(color), alpha] # convert color to RGBA
            mask_overlay=np.isin(masks-1, cell_indices)[..., np.newaxis]*color

        else: # multi-color mode
            num_labels=masks.max()+1 # number of unique labels (including background)
            mask_overlay=np.zeros((*masks.shape, 4))
            cell_colors=np.array([[*to_rgb(c), alpha] for c in cell_colors])
            color_map=np.zeros((num_labels, 4))
            for i, cell_index in enumerate(cell_indices):
                color_map[cell_index+1]=cell_colors[i]
            mask_overlay=color_map[masks]
        
        mask_overlay=np.rot90(mask_overlay, 3)
        mask_overlay=np.fliplr(mask_overlay)

        opaque_mask=mask_overlay.copy()
        opaque_mask[mask_overlay[...,-1]!=0, -1]=1

        layer[0].setImage(mask_overlay)
        layer[1].setImage(opaque_mask)
        return mask_overlay, opaque_mask

    def clear_selection_overlay(self):
        self.selection_overlay[0].clear()
        self.selection_overlay[1].clear()

    def clear_mask_overlay(self):
        self.mask_overlay[0].clear()
        self.mask_overlay[1].clear()


    def get_plot_coords(self, pos, pixels=True):
        """Get the pixel coordinates of the mouse cursor."""
        mouse_point = self.img_plot.plotItem.vb.mapSceneToView(pos) # axes are the same for both plots so we can use either to transform
        x, y = mouse_point.x(), mouse_point.y()
        if pixels:
            x, y = int(y), int(x)
        return x, y
    
    def mouse_moved(self, pos):
        x,y=self.get_plot_coords(pos, pixels=False)
        self.update_cursor(x, y)

        self.parent.mouse_moved(pos)

    def update_cursor(self, x, y):
        """Update the segmentation plot cursor based on the image plot cursor."""
        #if self.img_plot.sceneBoundingRect().contains(pos):
        #    mouse_point = self.img_plot.plotItem.vb.mapSceneToView(pos)
        #elif self.seg_plot.sceneBoundingRect().contains(pos):
        #    mouse_point = self.seg_plot.plotItem.vb.mapSceneToView(pos)
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

        # update segmentation overlay
        self.overlay_outlines()

        # update masks overlay, use the stored overlay if available
        if hasattr(self.parent.frame, 'mask_overlay') and self.parent.masks_checkbox.isChecked():
            img_masks, seg_masks=self.parent.frame.mask_overlay
            self.mask_overlay[0].setImage(img_masks)
            self.mask_overlay[1].setImage(seg_masks)
        else:
            self.overlay_masks()

        self.img.setImage(self.img_data)
        self.seg.setImage(self.seg_data)

    def rescale_handles(self, view_box):
        """Rescale the mask overlay when the view box changes."""
        if self.parent.file_loaded:
            from PyQt6.QtCore import QPointF
            point_scaling=self.img_plot.plotItem.vb.mapViewToScene(QPointF(1, 0))-self.img_plot.plotItem.vb.mapViewToScene(QPointF(0, 0))
            pixel_size = point_scaling.x()

            #for handle in self.parent.cell_roi.getHandles():
                #handle.radius = self.parent.cell_roi.handle_size*pixel_size
                #handle.buildPath()
                #handle.update()

    def sync_img_plot(self, view_box):
        """Sync the image plot view range with the segmentation plot."""
        self.rescale_handles(view_box)

        self.img_plot.blockSignals(True)
        self.img_plot.setRange(xRange=self.seg_plot.viewRange()[0], yRange=self.seg_plot.viewRange()[1], padding=0)
        self.img_plot.blockSignals(False)

    def sync_seg_plot(self, view_box):
        """Sync the segmentation plot view range with the image plot."""
        self.rescale_handles(view_box)

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

class CellMaskPolygon(QGraphicsPolygonItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPen(QPen(QColor(255, 255, 255, 100)))
        self.setBrush(QBrush(QColor(0, 255, 0, 100)))
        self.points=[]

    def clearPoints(self):
        self.points=[]
        self.update_polygon()

    def update_polygon(self):
        polygon=QPolygonF(self.points)
        self.setPolygon(polygon)

    def add_handle(self, y, x):
        y, x = y+0.5, x+0.5
        self.points.append(QPointF(y, x))
        self.update_polygon()
        self.last_handle_pos = (y, x)

    def get_enclosed_pixels(self):
        points=[(p.y(), p.x()) for p in self.points]
        shapely_polygon=Polygon(points).buffer(0.6)
        xmin, ymin, xmax, ymax=shapely_polygon.bounds
        x, y = np.meshgrid(np.arange(int(xmin), int(xmax)+1), np.arange(int(ymin), int(ymax)+1))

        bbox_grid=np.vstack((x.flatten(), y.flatten())).T
        enclosed_pixels=np.array([tuple(p) for p in bbox_grid if shapely_polygon.contains(Point(p))])
        
        return enclosed_pixels

class MainWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.file_loaded = False
        self.setWindowTitle("PyQtGraph Segmentation Viewer")
        self.resize(1080, 540)

        self.file_loaded = False

        # ----------------Toolbar items----------------
        self.spacer = QSpacerItem(20, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # RGB
        RGB_checkbox_widget = QWidget()
        self.RGB_layout = QHBoxLayout(RGB_checkbox_widget)
        self.RGB_checkboxes = [QCheckBox(s, self) for s in ['R', 'G', 'B']]
        for checkbox in self.RGB_checkboxes:
            checkbox.setChecked(True)
            self.RGB_layout.addWidget(checkbox)
        self.grayscale=QCheckBox("Grayscale", self)

        # Segmentation Overlay
        segmentation_overlay_widget = QWidget()
        segmentation_overlay_layout = QHBoxLayout(segmentation_overlay_widget)
        self.masks_checkbox = QCheckBox("Masks [X]", self)
        self.outlines_checkbox = QCheckBox("Outlines [Z]", self)
        segmentation_overlay_layout.addWidget(self.masks_checkbox)
        segmentation_overlay_layout.addWidget(self.outlines_checkbox)

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

        # FUCCI
        self.FUCCI_checkbox = QCheckBox("Show FUCCI Channel", self)
        self.cc_overlay_dropdown = QComboBox(self)
        self.cc_overlay_dropdown.addItems(["None", "Green", "Red", "All"])
        self.overlay_label = QLabel("FUCCI overlay", self)

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

        #----------------Frame Slider----------------
        self.frame_slider=QSlider(Qt.Orientation.Horizontal, self)
        # Customize the slider to look like a scroll bar
        self.frame_slider.setFixedHeight(15)  # Make the slider shorter in height
        self.frame_slider.setTickPosition(QSlider.TickPosition.NoTicks)  # No tick marks
        self.frame_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #888;
                border: 1px solid #444;
                height: 16px; /* Groove height */
                margin: 0px;
            }

            QSlider::handle:horizontal {
                background: #ccc;
                border: 1px solid #777;
                width: 40px; /* Handle width */
                height: 16px; /* Handle height */
                margin: -8px 0; /* Adjust positioning to align with groove */
                border-radius: 2px; /* Slightly round edges */
            }
        """)

        #----------------Layout----------------
        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 5, 5, 5)

        self.toolbar = QWidget()
        toolbar_layout = QVBoxLayout(self.toolbar)
        toolbar_layout.setSpacing(0)
        toolbar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.toolbar.setFixedWidth(175)

        self.canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(self.canvas_widget)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = PyQtGraphCanvas(parent=self)

        self.cell_roi = CellMaskPolygon()
        self.cell_roi.last_handle_pos = None
        self.canvas.img_plot.addItem(self.cell_roi)
        #self.canvas.img_plot.addItem(self.cell_roi.polygon_item)

        canvas_layout.addWidget(self.canvas)
        canvas_layout.addWidget(self.frame_slider)

        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.canvas_widget)

        tabbed_menu_widget = QTabWidget()
        tabbed_menu_widget.addTab(self.get_FUCCI_tab(), "FUCCI")
        tabbed_menu_widget.addTab(self.get_tracking_tab(), "Tracking")

        # Toolbar layout
        toolbar_layout.addWidget(RGB_checkbox_widget)
        toolbar_layout.addWidget(self.grayscale)

        toolbar_layout.addItem(self.spacer)

        toolbar_layout.addWidget(self.normalize_label)
        toolbar_layout.addWidget(self.normalize_widget)

        toolbar_layout.addItem(self.spacer)

        toolbar_layout.addWidget(segmentation_overlay_widget)

        toolbar_layout.addItem(self.spacer)
        
        toolbar_layout.addWidget(tabbed_menu_widget)
        
        toolbar_layout.addStretch() # spacer between top and bottom aligned widgets

        toolbar_layout.addWidget(self.command_line_button)

        toolbar_layout.addItem(self.spacer)


        toolbar_layout.addWidget(self.save_menu_widget)
        toolbar_layout.addWidget(self.save_stack)

        
        #----------------Connections----------------
        self.frame_slider.valueChanged.connect(self.update_frame_number)
        # click event
        self.canvas.img_plot.scene().sigMouseClicked.connect(self.on_click)
        self.canvas.seg_plot.scene().sigMouseClicked.connect(self.on_click)
        # RGB
        for checkbox in self.RGB_checkboxes:
            checkbox.stateChanged.connect(self.update_display)
        self.grayscale.stateChanged.connect(self.update_display)
        # normalize
        self.normalize_frame_button.toggled.connect(self.update_normalize_frame)
        self.command_line_button.clicked.connect(self.open_command_line)
        # segmentation overlay
        self.masks_checkbox.stateChanged.connect(self.canvas.overlay_masks)
        self.outlines_checkbox.stateChanged.connect(self.canvas.overlay_outlines)
        # FUCCI
        self.cc_overlay_dropdown.currentIndexChanged.connect(self.cc_overlay_changed)
        self.FUCCI_checkbox.stateChanged.connect(self.update_display)
        # tracking
        self.track_centroids_button.clicked.connect(self.track_centroids)
        self.tracking_range.returnPressed.connect(self.track_centroids)
        # save
        self.save_button.clicked.connect(self.save_segmentation)
        self.save_as_button.clicked.connect(self.save_as_segmentation)
    
    def update_frame_number(self, frame_number):
        if not self.file_loaded:
            return
        self.frame_number = frame_number
        self.imshow(self.stack.frames[self.frame_number], reset=False)

    def get_FUCCI_tab(self):
        FUCCI_tab = QWidget()
        FUCCI_layout = QVBoxLayout(FUCCI_tab)
        FUCCI_layout.setSpacing(0)
        FUCCI_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        FUCCI_layout.addWidget(self.overlay_label)
        FUCCI_layout.addWidget(self.cc_overlay_dropdown)
        FUCCI_layout.addWidget(self.FUCCI_checkbox)

        return FUCCI_tab
    
    def get_tracking_tab(self):
        tracking_tab = QWidget()
        tracking_layout = QVBoxLayout(tracking_tab)
        tracking_layout.setSpacing(0)
        tracking_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        range_label = QLabel('Search Range:  ', self)
        self.tracking_range = QLineEdit("10", self)
        self.tracking_range.setValidator(QIntValidator(bottom=1))
        self.track_centroids_button = QPushButton("Track Centroids", self)

        self.tracking_range_layout=QHBoxLayout()
        self.tracking_range_layout.addWidget(range_label)
        self.tracking_range_layout.addWidget(self.tracking_range)
        tracking_layout.addLayout(self.tracking_range_layout)
        tracking_layout.addWidget(self.track_centroids_button)
        return tracking_tab

    def update_coordinate_label(self, x, y):
        self.status_coordinates.setText(f"Coordinates: ({x}, {y})")
    
    def update_cell_label(self, cell_n):
        if cell_n is None:
            self.status_cell.setText("Selected Cell: None")
        else:
            self.status_cell.setText(f"Selected Cell: {cell_n}")

    def track_centroids(self):
        if not self.file_loaded:
            return

        tracking_range=self.tracking_range.text()
        if tracking_range == '':
            return
        
        self.stack.track_centroids(search_range=int(tracking_range))
        print(f'Tracked centroids for stack {self.stack.name}')

        # recolor cells so each particle has one color over time
        for frame in self.stack.frames:
            if hasattr(frame, 'mask_overlay'):
                del frame.mask_overlay # remove the mask_overlay attribute to force recoloring
            
        t=self.stack.tracked_centroids
        colors=np.random.randint(10, size=t['particle'].nunique())
        # create a t['color'] column
        t['color']=colors[t['particle']]

        from matplotlib import colormaps
        cmap=colormaps.get_cmap('tab10')
        for frame in self.stack.frames:
            tracked_frame=t[t.frame==frame.frame_number].sort_values('cell_number')
            frame.set_cell_attr('color_ID', cmap(tracked_frame['color']))

        self.canvas.overlay_masks()

    def update_display(self):
        """Redraw the image data with whatever new settings have been applied from the toolbar."""
        if self.file_loaded:
            img_data=self.normalize(self.frame.img)
            self.canvas.update_display(img_data=img_data, seg_data=self.frame.outlines)
    
    def update_normalize_frame(self):
        self.normalize_stack=self.normalize_stack_button.isChecked()
        self.update_display()

    def normalize(self, img):
        if self.frame.img.ndim==2: # single channel
            colors=1
        else:
            colors=3
        if self.normalize_stack and self.stack is not None: # normalize the stack
            if hasattr(self.stack, 'bounds'):
                bounds=self.stack.bounds
            else:
                all_imgs=np.array([frame.img for frame in self.stack.frames]).reshape(-1, colors)
                bounds=np.quantile(all_imgs, (0.01, 0.99), axis=0).T
                self.stack.bounds=bounds

            normed_img=preprocessing.normalize(img, bounds=bounds)
        else: # normalize the frame
            if hasattr(self.frame, 'bounds'):
                bounds=self.frame.bounds
            else:
                bounds=np.quantile(img.reshape(-1,colors), (0.01, 0.99), axis=0).T
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
        overlay_color=self.cc_overlay_dropdown.currentText().lower()

        if event.button() == Qt.MouseButton.RightButton and overlay_color=='none': # segmentation
            if not self.drawing_cell_roi:
                self.drawing_cell_roi=True
                self.cell_roi.points=[]

                x, y = self.canvas.get_plot_coords(event.scenePos(), pixels=True)
                # Add the first handle
                self.cell_roi.add_handle(y, x)
                self.canvas.rescale_handles(self.canvas.img_plot.getViewBox())
                self.cell_roi.first_handle_pos=np.array((y, x))
                self.cell_roi.last_handle_pos=np.array((y, x))

                self.roi_is_closeable=False
                
            else:
                self.close_roi()

        elif current_cell_n>=0:
            if overlay_color=='none': # basic selection
                if event.button() == Qt.MouseButton.LeftButton:
                    if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                        # ctrl click deletes if a mask overlay is enabled
                        if self.masks_checkbox.isChecked() or self.outlines_checkbox.isChecked():
                            self.delete_cell_mask(current_cell_n)
                            self.selected_cell=None
                            self.selected_particle=None
                    elif current_cell_n==self.selected_cell:
                        # clicking the same cell again deselects it
                        self.selected_cell=None
                        self.selected_particle=None
                    else:
                        # select the cell
                        self.selected_cell=current_cell_n
                    self.canvas.highlight_cells([self.selected_cell], alpha=0.3, color='white')

            else: # cell cycle classification
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
            self.selected_particle=None
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

    def mouse_moved(self, pos):
        if self.file_loaded and self.drawing_cell_roi:
            x, y = self.canvas.get_plot_coords(pos, pixels=True) # position in plot coordinates
            
            if np.array_equal((y, x), self.cell_roi.last_handle_pos):
                return
            else:
                self.cell_roi.add_handle(y, x)
                if self.roi_is_closeable:
                    if np.linalg.norm(np.array((y, x))-self.cell_roi.first_handle_pos)<3:
                        self.close_roi()
                        return
                else:
                    if np.linalg.norm(np.array((y, x))-self.cell_roi.first_handle_pos)>3:
                        self.roi_is_closeable=True
                    
                self.canvas.rescale_handles(self.canvas.img_plot.getViewBox())
        
    def get_cell(self, x, y):
        if x < 0 or y < 0 or x >= self.canvas.img_data.shape[1] or y >= self.canvas.img_data.shape[0]:
            return -1 # out of bounds
        cell_n=self.frame.masks[x, y]-1
        return cell_n
    
    def close_roi(self):
        self.drawing_cell_roi=False
        enclosed_pixels=self.cell_roi.get_enclosed_pixels()
        self.add_cell_mask(enclosed_pixels)
        self.cell_roi.clearPoints()
        self.update_display()

    def random_color(self, n_samples=None, cmap='tab10', n_colors=10):
        from matplotlib import colormaps
        random_colors=np.random.randint(0, n_colors, size=n_samples)
        cmap=colormaps.get_cmap(cmap)
        colors=np.array(cmap(random_colors))[...,:3]

        return colors
        
    def add_cell_mask(self, enclosed_pixels):
        from cellpose import utils
        new_mask_n=self.frame.n+1
        cell_mask=np.zeros_like(self.frame.masks, dtype=bool)
        cell_mask[enclosed_pixels[:,0], enclosed_pixels[:,1]]=True
        new_mask=cell_mask & (self.frame.masks==0)
        self.new_mask=new_mask
        if new_mask.sum()>3: # if the mask is larger than 3 pixels
            self.frame.masks[new_mask]=new_mask_n
            self.frame.n=new_mask_n
            print(f'Added cell {new_mask_n}')
            print(f'added {new_mask.sum()} pixels')

            cell_color=self.random_color()
            outline=utils.outlines_list(new_mask)[0]

            self.frame.outlines[outline[:,1], outline[:,0]]=True
            self.frame.cells=np.append(self.frame.cells, Cell(new_mask_n, outline, color_ID=cell_color, red=False, green=False, frame_number=self.frame.frame_number))
            
            self.canvas.overlay_masks()
            self.canvas.overlay_outlines()
            return True
        else:
            return False
    
    def delete_cell_mask(self, cell_n):
        from monolayer_tracking.preprocessing import renumber_masks
        to_clear=self.frame.masks==cell_n+1
        self.frame.masks[to_clear]=0
        self.frame.outlines[to_clear]=False
        print(f'Deleted cell {cell_n}')
        print(f'deleted {to_clear.sum()} pixels')
        
        self.frame.cells=np.delete(self.frame.cells, cell_n)

        self.frame.masks=renumber_masks(self.frame.masks)
        self.frame.n=self.frame.masks.max()

        for n, cell in enumerate(self.frame.cells):
            cell.n=n

        self.canvas.overlay_masks()
        self.update_display()

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

    def cc_overlay_changed(self):
        if not self.file_loaded:
            return
        
        overlay_color=self.cc_overlay_dropdown.currentText().lower()

        # set RGB mode
        if overlay_color == 'none':
            self.canvas.set_RGB(True)
        else:
            if overlay_color == 'all':
                self.canvas.set_RGB([True, True, False])
            elif overlay_color == 'red':
                self.canvas.set_RGB([True, False, False])
            elif overlay_color == 'green':
                self.canvas.set_RGB([False, True, False])
            
            # set overlay mode
            self.outlines_checkbox.setChecked(True)
            self.masks_checkbox.setChecked(False)
        
        self.cc_overlay()

    def cc_overlay(self, event=None):
        """Handle cell cycle overlay options."""
        overlay_color=self.cc_overlay_dropdown.currentText().lower()
        if overlay_color == 'none': # clear overlay
            self.selected_cell=None
            self.selected_particle=None
            self.update_cell_label(None)
            self.canvas.highlight_cells([])
        else:
            if overlay_color == 'all':
                colors=np.array(['g','r','orange'])
                green, red=np.array(self.frame.get_cell_attr(['green', 'red'])).T
                colored_cells=np.where(red | green)[0] # cells that are either red or green
                cell_cycle=green+2*red-1
                cell_colors=colors[cell_cycle[colored_cells]] # map cell cycle state to green, red, orange
                self.canvas.highlight_cells(colored_cells, alpha=0.1, cell_colors=cell_colors)

            else:
                colored_cells=np.where(self.frame.get_cell_attr(overlay_color))[0]
                self.canvas.highlight_cells(colored_cells, alpha=0.1, color=overlay_color)

    # Drag and drop event
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def reset_display(self):
        self.drawing_cell_roi=False
        self.selected_cell=None
        self.selected_particle=None
        self.update_cell_label(None)
        self.cc_overlay_dropdown.setCurrentIndex(0) # clear overlay
        self.canvas.set_RGB(True)
        self.grayscale.setChecked(False)
        self.canvas.clear_selection_overlay() # remove any overlays (highlighting, outlines)
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
            # preserve selected cell if tracking info is available
            if hasattr(self, 'selected_particle') and self.selected_particle is not None:
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
                self.canvas.highlight_cells([self.selected_cell], alpha=0.3, color='white', layer='selection')
            
            # or clear highlight
            else:
                self.canvas.clear_selection_overlay() # no tracking data, clear highlights

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
        self.open_stack(natsorted(files))

    def open_stack(self, files):
        self.load_files(files)
        if not self.file_loaded:
            return
        self.frame_number = 0
        self.imshow(self.stack.frames[self.frame_number])
        self.canvas.img_plot.autoRange()
        self.globals_dict['stack']=self.stack
        self.frame_slider.setRange(0, len(self.stack.frames)-1)
        

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

        # segmentation overlay
        if event.key() == Qt.Key.Key_X:
            self.masks_checkbox.toggle()
        elif event.key() == Qt.Key.Key_Z:
            self.outlines_checkbox.toggle()

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
        layout = QVBoxLayout(self)
        
        # Terminal-style output display area (Read-Only)
        self.terminal_display = QTextEdit(self)
        self.terminal_display.setStyleSheet("""
            background-color: black;
            color: white;
            font-family: "Courier";
            font-size: 10pt;
        """)
        self.terminal_display.setReadOnly(True)
        layout.addWidget(self.terminal_display)
        
        # Command input area
        self.command_input = QLineEdit(self)
        self.command_input.setStyleSheet("""
            background-color: black;
            color: white;
            font-family: "Courier";
            font-size: 12pt;
        """)
        #self.command_input.setPlaceholderText("Type your command here and press Enter")
        layout.addWidget(self.command_input)

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

if __name__ == '__main__':
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    app.setStyleSheet(dark_mode_style)
    app.quitOnLastWindowClosed = True
    ui = MainWidget()
    ui.show()
    app.exec()