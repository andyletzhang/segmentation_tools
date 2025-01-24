import sys
import numpy as np
import pandas as pd
from cellpose import utils # takes a while to import :(
import os
import fastremap
from scipy import ndimage
from skimage import draw

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QComboBox, QPushButton, QRadioButton, QInputDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QCheckBox, QSpacerItem, QSizePolicy, QFileDialog, QDialog,
    QLineEdit, QTabWidget, QSlider, QGraphicsEllipseItem, QFormLayout, QSplitter, QProgressBar, QScrollArea,
    QTableWidgetItem, QTableWidget
)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QIntValidator, QDoubleValidator, QIcon, QFontMetrics, QMouseEvent
from superqt import QRangeSlider
import pyqtgraph as pg

from segmentation_tools.segmented_comprehension import SegmentedStack, Cell
from segmentation_tools.io import segmentation_from_img, segmentation_from_zstack
from segmentation_viewer.canvas import PyQtGraphCanvas, CellMaskPolygons, CellSplitLines
from segmentation_viewer.command_line import CommandLineWindow
from segmentation_viewer.qt import CustomComboBox, FineScrubQRangeSlider, SubstackDialog
from segmentation_viewer.io import ExportWizard

from natsort import natsorted
import importlib.resources
from pathlib import Path
from tqdm import tqdm

# high priority
# TODO: frame histogram should have options for aggregating over frame or stack
# TODO: import masks (and everything else except img/zstack)
# TODO: File -> export heights tif, import heights tif
# TODO: fix segmentation stat LUTs, implement stack LUTs (when possible). Allow floats when appropriate

# low priority
# TODO: ctrl+shift+click to delete particle
# TODO: when cell is clicked, have option to show its entire colormapped track
# TODO: get_mitoses, visualize mitoses, edit mitoses
# TODO: use fastremap to add cell highlights?

# TODO: some image pyramid approach to speed up work on large images??
# TODO: maybe load images/frames only when they are accessed? (lazy loading)
# TODO: number of neighbors

# eventual QOL improvements
# TODO: rename and streamline update_display, imshow. Combine other updates (plot_particle_statistic etc.)
# TODO: normalize the summed channels when show_grayscale
# TODO: make sure all frames have same number of z slices
# TODO: perhaps allow for non-contiguous masks and less numerical reordering.
    # 1. Replace masks.max() with n_cells, np.unique(masks) everywhere
    # 2. Iterate over unique IDs instead of range(n_cells), and/or skip empty IDs
    # 3. Rewrite delete cell to remove mask ID without renumbering

# TODO: undo/redo
# TODO: add mouse and keyboard shortcuts to interface
# TODO: FUCCI tab - show cc occupancies as a stacked bar
# TODO: expand/collapse segmentation plot
# TODO: pick better colors for highlight track ends which don't overlap with FUCCI
# TODO: user can specify membrane channel for volumes tab
# TODO: mask nan slices during normalization

class MainWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        # window setup
        self.setWindowTitle("Segmentation Viewer")
        icon_path=importlib.resources.files('segmentation_viewer.assets').joinpath('icon.png')
        self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1280, 720)
        self.file_loaded = False # passive mode
        self.is_grayscale = False
        self.drawing_cell_roi = False
        self.drawing_cell_split = False
        self.spacer = (0,10) # default spacer size (width, height)
        self.globals_dict = {'main': self, 'np': np}
        self.locals_dict = {}
        self.font_metrics=QFontMetrics(QLabel().font()) # metrics for the default font
        self.digit_width=self.font_metrics.horizontalAdvance('0') # text length scale
        self.cancel_iter=False # flag to cancel progress bar iteration

        # Menu bar
        from segmentation_viewer.qt import create_action
        self.menu_bar = self.menuBar()
        
        # FILE
        self.file_menu = self.menu_bar.addMenu("File")
        self.file_menu.addAction(create_action("Open File(s)", self.open_files, self, 'Ctrl+O'))
        self.file_menu.addAction(create_action("Open Folder", self.open_folder_dialog, self, 'Ctrl+Shift+O'))
        self.file_menu.addAction(create_action("Save", self.save_segmentation, self, 'Ctrl+S'))
        self.file_menu.addAction(create_action("Save As", self.save_as_segmentation, self, 'Ctrl+Shift+S'))
        self.file_menu.addAction(create_action("Export CSV...", self.export_csv, self, 'Ctrl+Shift+E'))
        self.file_menu.addAction(create_action("Import Image(s)", self.import_images, self))
        self.file_menu.addAction(create_action("Exit", self.close, self, 'Ctrl+Q'))
        #self.file_menu.addAction(create_action("Import Masks...", self.import_masks, self))

        # EDIT
        self.edit_menu = self.menu_bar.addMenu("Edit")
        #self.edit_menu.addAction(create_action("Undo", self.undo, self, 'Ctrl+Z'))
        #self.edit_menu.addAction(create_action("Redo", self.redo, self, 'Ctrl+Shift+Z'))
        self.edit_menu.addAction(create_action("Clear Masks", self.clear_masks, self))
        self.edit_menu.addAction(create_action("Generate Outlines", self.generate_outlines_list, self))
        self.edit_menu.addAction(create_action("Mend Gaps", self.mend_gaps, self))
        self.edit_menu.addAction(create_action("Remove Edge Masks", self.remove_edge_masks, self))

        # VIEW
        self.view_menu = self.menu_bar.addMenu("View")
        self.view_menu.addAction(create_action("Reset View", self.reset_view, self))
        self.view_menu.addAction(create_action("Show Grayscale", self.toggle_grayscale, self))
        self.view_menu.addAction(create_action("Overlay Settings...", self.open_overlay_settings, self))
        #self.view_menu.addAction(create_action("Segmentation Plot", self.toggle_segmentation_plot, self))

        # IMAGE
        self.image_menu = self.menu_bar.addMenu("Image")
        self.image_menu.addAction(create_action("Reorder Channels", self.reorder_channels, self))
        #self.image_menu.addAction(create_action("Set Voxel Size", self.voxel_size_prompt, self))

        # STACK
        self.stack_menu = self.menu_bar.addMenu("Stack")
        self.stack_menu.addAction(create_action("Delete frame", self.delete_frame, self))
        self.stack_menu.addAction(create_action("Make substack...", self.make_substack, self))

        # HELP
        self.help_menu = self.menu_bar.addMenu("Help")
        self.help_menu.addAction(create_action("Pull updates", self.update_packages, self))

        # Status bar
        self.status_cell=QLabel("Selected Cell: None", self)
        self.status_frame_number=QLabel("Frame: None", self)
        self.status_tracking_ID=QLabel("Tracking ID: None", self)
        self.status_coordinates=QLabel("Cursor: (x, y)", self)
        self.status_pixel_value=QLabel("R: None, G: None, B: None", self)
        self.statusBar().addWidget(self.status_cell)
        self.statusBar().addWidget(self.status_frame_number)
        self.statusBar().addWidget(self.status_tracking_ID)
        self.statusBar().addPermanentWidget(self.status_coordinates)
        self.statusBar().addPermanentWidget(self.status_pixel_value)

        #----------------Frame Slider----------------
        self.frame_slider=QSlider(Qt.Orientation.Horizontal, self)
        self.zstack_slider=QSlider(Qt.Orientation.Vertical, self)
        self.frame_slider.setVisible(False)
        self.zstack_slider.setVisible(False)
        # Customize the slider to look like a scroll bar
        self.frame_slider.setFixedHeight(15)  # Make the slider shorter in height
        self.frame_slider.setTickPosition(QSlider.TickPosition.NoTicks)  # No tick marks
        
        self.zstack_slider.setFixedWidth(15)  # Make the slider shorter in height
        self.zstack_slider.setTickPosition(QSlider.TickPosition.NoTicks)  # No tick marks

        #----------------Layout----------------
        # Main layout
        main_widget = QSplitter()
        self.setCentralWidget(main_widget)

        self.canvas_widget = QWidget()
        canvas_VBoxLayout = QVBoxLayout(self.canvas_widget)
        canvas_VBoxLayout.setSpacing(0)
        canvas_VBoxLayout.setContentsMargins(0, 0, 0, 0)
        canvas_HBoxLayout = QHBoxLayout()
        canvas_HBoxLayout.setSpacing(0)
        canvas_HBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.canvas = PyQtGraphCanvas(parent=self)
        self.globals_dict['canvas']=self.canvas

        self.cell_roi = CellMaskPolygons()
        self.cell_roi.last_handle_pos = None
        self.canvas.img_plot.addItem(self.cell_roi.img_poly)
        self.canvas.seg_plot.addItem(self.cell_roi.seg_poly)

        self.cell_split = CellSplitLines()
        self.canvas.img_plot.addItem(self.cell_split.img_line)
        self.canvas.seg_plot.addItem(self.cell_split.seg_line)
        
        self.right_toolbar=self.get_right_toolbar()
        self.left_toolbar=self.get_left_toolbar()

        canvas_HBoxLayout.addWidget(self.canvas)
        canvas_HBoxLayout.addWidget(self.zstack_slider)
        canvas_VBoxLayout.addLayout(canvas_HBoxLayout)
        canvas_VBoxLayout.addWidget(self.frame_slider)

        main_widget.addWidget(self.left_toolbar)
        main_widget.addWidget(self.canvas_widget)
        main_widget.addWidget(self.right_toolbar)
        main_widget.setCollapsible(1, False)
        main_widget.setSizes([250, 800, 250])

        self.default_visual_settings=self.visual_settings
        self.default_visual_settings['LUTs']=None
        self.saved_visual_settings=[self.default_visual_settings for _ in range(4)]
        self.current_tab=0
        self.FUCCI_mode=False

        #----------------Connections----------------
        self.frame_slider.valueChanged.connect(self.change_current_frame)
        self.zstack_slider.valueChanged.connect(self.update_zstack_number)

        # click event
        self.canvas.img_plot.scene().sigMouseClicked.connect(self.on_click)
        self.canvas.seg_plot.scene().sigMouseClicked.connect(self.on_click)
    
    def get_right_toolbar(self):
        self.stat_tabs=QTabWidget()
        self.stat_tabs.addTab(self.get_histogram_tab(), "Histogram")
        self.stat_tabs.addTab(self.get_particle_stat_tab(), "Particle")

        stat_overlay_widget=QWidget(objectName='bordered')
        stat_overlay_layout=QVBoxLayout(stat_overlay_widget)
        stat_overlay_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        seg_overlay_layout=QHBoxLayout()
        self.seg_overlay_label=QLabel("Overlay Statistic:", self)
        self.seg_overlay_attr=CustomComboBox(self)
        self.seg_overlay_attr.addItems(['Select Cell Attribute'])
        seg_overlay_layout.addWidget(self.seg_overlay_label)
        seg_overlay_layout.addWidget(self.seg_overlay_attr)

        normalize_label = QLabel("Overlay LUTs:", self)
        normalize_widget=QWidget()
        normalize_layout=QHBoxLayout(normalize_widget)
        normalize_layout.setContentsMargins(0, 0, 0, 0)
        self.stat_frame_button=QRadioButton("Frame", self)
        self.stat_stack_button=QRadioButton("Stack", self)
        self.stat_custom_button=QRadioButton("LUT", self)
        normalize_layout.addWidget(self.stat_frame_button)
        normalize_layout.addWidget(self.stat_stack_button)
        normalize_layout.addWidget(self.stat_custom_button)
        self.stat_frame_button.setChecked(True)
        self.stat_LUT_type='frame'
        slider_layout, self.stat_LUT_slider, self.stat_range_labels=self.labeled_LUT_slider(default_range=(0, 255))

        cell_ID_widget=QWidget(objectName='bordered')
        self.cell_ID_layout=QFormLayout(cell_ID_widget)
        selected_cell_label=QLabel("Cell ID:", self)
        self.selected_cell_prompt=QLineEdit(self, placeholderText='None')
        self.selected_cell_prompt.setValidator(QIntValidator(bottom=0)) # non-negative integers only
        selected_particle_label=QLabel("Tracking ID:", self)
        self.selected_particle_prompt=QLineEdit(self, placeholderText='None')
        self.selected_particle_prompt.setValidator(QIntValidator(bottom=0)) # non-negative integers only
        self.cell_properties_label=QLabel(self)
        self.cell_ID_layout.addRow(selected_cell_label, self.selected_cell_prompt)
        self.cell_ID_layout.addRow(selected_particle_label, self.selected_particle_prompt)
        self.cell_ID_layout.addRow(self.cell_properties_label)

        stat_overlay_layout.addLayout(seg_overlay_layout)
        stat_overlay_layout.addWidget(normalize_label)
        stat_overlay_layout.addWidget(normalize_widget)
        stat_overlay_layout.addLayout(slider_layout)

        # Create a container widget for all content
        particle_stat_layout=QSplitter(Qt.Orientation.Vertical)
        particle_stat_layout.setContentsMargins(5,10,10,10)
        particle_stat_layout.addWidget(self.stat_tabs)
        particle_stat_layout.addWidget(stat_overlay_widget)
        particle_stat_layout.addWidget(cell_ID_widget)
        particle_stat_layout.setSizes([200, 300, 200])

        # Set up scroll area
        right_scroll_area = QScrollArea()
        right_scroll_area.setWidgetResizable(True)
        right_scroll_area.setWidget(particle_stat_layout)
        right_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll_area.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)
        right_scroll_area.setMinimumWidth(250)

        #----connections-----
        # cell selection
        self.selected_cell_prompt.textChanged.connect(self.cell_prompt_changed)
        self.selected_cell_prompt.returnPressed.connect(self.cell_prompt_changed)
        self.selected_particle_prompt.textChanged.connect(self.particle_prompt_changed)
        self.selected_particle_prompt.returnPressed.connect(self.particle_prompt_changed)
        # stat overlay
        self.seg_overlay_attr.dropdownOpened.connect(self.get_overlay_attrs)
        self.seg_overlay_attr.activated.connect(self.new_seg_overlay)
        self.seg_overlay_attr.currentIndexChanged.connect(self.new_seg_overlay)
        self.stat_LUT_slider.valueChanged.connect(self.stat_LUT_slider_changed)
        self.stat_frame_button.toggled.connect(self.update_stat_LUT)
        self.stat_stack_button.toggled.connect(self.update_stat_LUT)
        self.stat_custom_button.toggled.connect(self.update_stat_LUT)
        return right_scroll_area

    def get_histogram_tab(self):
        frame_histogram_widget=QWidget()
        frame_histogram_layout=QVBoxLayout(frame_histogram_widget)
        histogram_menu_layout=QHBoxLayout()
        self.histogram_menu=CustomComboBox(self)
        self.histogram_menu.addItems(['Select Cell Attribute'])
        histogram_menu_layout.addWidget(self.histogram_menu)
        histogram_menu_layout.setContentsMargins(40, 0, 0, 0) # indent the title/menu
        self.histogram=pg.PlotWidget(background='transparent')
        self.histogram.setMinimumHeight(200)
        self.histogram.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.histogram.setLabel('bottom', 'Select Cell Attribute')
        self.histogram.setLabel('left', 'Probability Density')
        self.histogram.showGrid(x=True, y=True)

        frame_histogram_layout.addLayout(histogram_menu_layout)
        frame_histogram_layout.addWidget(self.histogram)

        self.histogram_menu.dropdownOpened.connect(self.get_histogram_attrs)
        self.histogram_menu.activated.connect(self.new_histogram)
        self.histogram_menu.currentIndexChanged.connect(self.new_histogram)
        return frame_histogram_widget

    def stat_LUT_slider_changed(self):
        self.stat_custom_button.blockSignals(True)
        self.stat_custom_button.setChecked(True)
        self.stat_custom_button.blockSignals(False)
        self.set_stat_LUT_levels(self.stat_LUT_slider.value())

    def set_stat_LUT_levels(self, levels):
        # TODO: RuntimeWarning: invalid value encountered in cast data=data.astype(int) at level=256, only when working with a stack
        if levels[0]==np.nan and levels[1]==np.nan:
            levels=(0,1)
        self.canvas.seg_stat_overlay.setLevels(levels)
        self.stat_LUT_slider.blockSignals(True)
        self.stat_LUT_slider.setValue(levels)
        self.stat_LUT_slider.blockSignals(False)
        self.canvas.cb.setLevels(levels) # TODO: better colorbar tick labels

    def update_stat_LUT(self):
        if self.stat_frame_button.isChecked():
            self.stat_LUT_type='frame'
        elif self.stat_stack_button.isChecked():
            self.stat_LUT_type='stack'
        else:
            self.stat_LUT_type='custom'
        
        self.show_seg_overlay()

    def cell_scalar_attrs(self, cell):
        ''' Return all common attributes which are scalar cell attributes '''
        attrs=set(dir(cell))
        ignored_attrs={'red','green','vertex_area','shape_parameter','sorted_vertices','vertex_perimeter'}
        test_attrs=attrs-ignored_attrs
        # Collect attributes to remove instead of modifying the set in place
        to_remove = set()
        for attr in test_attrs:
            if attr.startswith('_'):
                to_remove.add(attr)
                continue
            try:
                val = getattr(cell, attr)
            except AttributeError:
                to_remove.add(attr)
                continue
            if not np.isscalar(val):
                to_remove.add(attr)

        # Remove collected attributes from test_attrs
        test_attrs -= to_remove
        return test_attrs
    
    def cell_stat_attrs(self, cell):
        ''' Return all common attributes which are meaningful cell-level metrics '''
        ignored_attrs={'cycle_stage','n','frame'}
        attrs=self.cell_scalar_attrs(cell)-ignored_attrs

        return attrs
    
    def get_cell_frame_attrs(self):
        items=[]
        if len(self.frame.cells)>0: # add any valid scalar cell attributes
            common_attrs=self.cell_stat_attrs(self.frame.cells[0])
            for cell in self.frame.cells[1:]:
                common_attrs=common_attrs.intersection(set(dir(cell)))
            items.extend(list(common_attrs))

        return items
    
    def get_histogram_attrs(self):
        if not self.file_loaded:
            return
        current_attr=self.histogram_menu.currentText()
        items=self.get_cell_frame_attrs()
        items=['Select Cell Attribute']+natsorted(items)
        self.histogram_menu.blockSignals(True)
        self.histogram_menu.clear()
        self.histogram_menu.addItems(items)
        self.histogram_menu.blockSignals(False)
        current_index=self.histogram_menu.findText(current_attr)
        if current_index==-1:
            current_index=0
        self.histogram_menu.setCurrentIndex(current_index)

    def get_overlay_attrs(self):
        if not self.file_loaded:
            return
        current_attr=self.seg_overlay_attr.currentText()
        items=self.get_cell_frame_attrs()
        if hasattr(self.frame, 'heights'):
            items.append('heights')
        items=['Select Cell Attribute']+natsorted(items)
        self.seg_overlay_attr.blockSignals(True)
        self.seg_overlay_attr.clear()
        self.seg_overlay_attr.addItems(items)
        self.seg_overlay_attr.blockSignals(False)
        current_index=self.seg_overlay_attr.findText(current_attr)
        if current_index==-1:
            current_index=0
        self.seg_overlay_attr.setCurrentIndex(current_index)

    def new_seg_overlay(self):
        # TODO: adapt LUT range slider to accept floats
        if not self.file_loaded:
            return
        plot_attr=self.seg_overlay_attr.currentText()

        if plot_attr=='Select Cell Attribute' or plot_attr is None:
            self.stat_LUT_slider.blockSignals(True)
            self.stat_LUT_slider.setRange(0, 255)
            self.stat_LUT_slider.setValue((0, 255))
            self.stat_LUT_slider.blockSignals(False)
            self.clear_seg_stat()
            return
        
        else:
            if plot_attr=='heights':
                stat=[]
                for frame in self.stack.frames:
                    if hasattr(frame, 'heights'):
                        stat.append(frame.heights.flatten())
                stat=np.concatenate(stat)
            else:
                cell_attrs=[]
                for frame in self.stack.frames:
                    try:
                        cell_attrs.extend(frame.get_cell_attrs(plot_attr))
                    except AttributeError:
                        continue
                stat=np.array(cell_attrs)
                if len(stat)==0:
                    print(f'Attribute {plot_attr} not found in cells')
                    return
            LUT_range=(np.nanmin(stat), np.nanmax(stat))
            self.stat_LUT_slider.blockSignals(True)
            self.stat_LUT_slider.setRange(*LUT_range)
            self.stat_LUT_slider.blockSignals(False)

            if self.stat_LUT_type=='custom': # change the LUT range to match the new data
                self.stat_frame_button.setChecked(True)

            self.show_seg_overlay()

    def show_seg_overlay(self, event=None):
        if not self.file_loaded:
            return
        plot_attr=self.seg_overlay_attr.currentText()
        if plot_attr=='Select Cell Attribute':
            self.canvas.cb.setVisible(False)
            self.clear_seg_stat()
        else:
            self.canvas.cb.setVisible(True)
            if plot_attr=='heights':
                if not hasattr(self.frame, 'heights'):
                    self.seg_overlay_attr.setCurrentIndex(0)
                    return
                
                if not hasattr(self.frame, 'z_scale'):
                    print(f'No z scale found for {self.frame.name}, defaulting to 1.')
                    self.z_size.setText('1.0')
                    self.update_voxel_size()

                self.overlay_seg_stat(self.frame.scaled_heights)
            else:
                cell_attrs=np.array(self.frame.get_cell_attrs(plot_attr, fill_value=np.nan))

                value_map=np.concatenate([[np.nan], cell_attrs.astype(float)])
                mask_values=value_map[self.frame.masks]
                self.overlay_seg_stat(mask_values)

    def overlay_seg_stat(self, stat=None):
        if not self.file_loaded:
            return
        if stat is None:
            stat=self.canvas.seg_stat_overlay.image
        if np.all(np.isnan(stat)):
            levels=(0, 1)
            self.canvas.seg_stat_overlay.clear()
        else:
            stat_range=(np.nanmin(stat), np.nanmax(stat))
            if self.stat_LUT_type=='frame':
                levels=(stat_range)
            elif self.stat_LUT_type=='stack':
                levels=(stat_range) # TODO: stack levels
            elif self.stat_LUT_type=='custom':
                levels=self.stat_LUT_slider.value()
            self.canvas.seg_stat_overlay.setImage(self.canvas.transform_image(stat))

        self.stat_LUT_slider.blockSignals(True)
        self.set_stat_LUT_levels(levels)
        self.stat_LUT_slider.blockSignals(False)

        self.stat_range_labels[0].setText(str(round(levels[0], 2)))
        self.stat_range_labels[1].setText(str(round(levels[1], 2)))

    def clear_seg_stat(self):
        self.canvas.seg_stat_overlay.clear()

    def get_particle_stat_tab(self):
        particle_plot_widget=QWidget()
        particle_plot_layout=QVBoxLayout(particle_plot_widget)
        particle_stat_menu_layout=QHBoxLayout()
        self.particle_stat_menu=CustomComboBox(self)
        particle_stat_menu_layout.addWidget(self.particle_stat_menu)
        particle_stat_menu_layout.setContentsMargins(40, 0, 0, 0) # indent the title/menu
        self.particle_stat_menu.addItem('Select Cell Attribute')
        self.particle_stat_plot=pg.PlotWidget(background='transparent')
        self.particle_stat_plot.setLabel('left', 'Select Cell Attribute')
        self.particle_stat_plot.setMinimumHeight(200)
        self.particle_stat_plot.setLabel('bottom', 'Frame')
        self.stat_plot_frame_marker=pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=2))
        self.particle_stat_plot.addItem(self.stat_plot_frame_marker)


        particle_plot_layout.addLayout(particle_stat_menu_layout)
        particle_plot_layout.addWidget(self.particle_stat_plot)

        # connect particle measurements
        self.particle_stat_menu.dropdownOpened.connect(self.get_particle_attrs)
        self.particle_stat_menu.currentIndexChanged.connect(self.plot_particle_statistic)

        return particle_plot_widget

    def get_left_toolbar(self):
        open_menu=QHBoxLayout()
        open_menu.setSpacing(5)

        # RGB
        self.RGB_checkbox_layout = QVBoxLayout()
        self.add_RGB_checkboxes(self.RGB_checkbox_layout)

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
        self.normalize_layout.setContentsMargins(0, 0, 0, 0)
        self.normalize_frame_button=QRadioButton("Frame", self)
        self.normalize_stack_button=QRadioButton("Stack", self)
        self.normalize_custom_button=QRadioButton("LUT", self)
        self.normalize_layout.addWidget(self.normalize_frame_button)
        self.normalize_layout.addWidget(self.normalize_stack_button)
        self.normalize_layout.addWidget(self.normalize_custom_button)
        self.normalize_type='frame'

        # LUTs
        self.slider_layout=QVBoxLayout()
        self.add_RGB_sliders(self.slider_layout)
        
        LUT_widget=QWidget(objectName='bordered')
        LUT_layout=QVBoxLayout(LUT_widget)
        LUT_layout.setSpacing(0)
        LUT_layout.addLayout(self.RGB_checkbox_layout)
        LUT_layout.addItem(self.vertical_spacer())
        LUT_layout.addWidget(self.normalize_label)
        LUT_layout.addWidget(self.normalize_widget)
        LUT_layout.addLayout(self.slider_layout)
        LUT_layout.addWidget(segmentation_overlay_widget)

        # Voxel size
        self.voxel_size_widget=QWidget(objectName='bordered')
        self.voxel_size_VLayout=QVBoxLayout(self.voxel_size_widget)
        self.voxel_size_HLayout=QHBoxLayout()
        voxel_size_label=QLabel("Voxel Size (μm):", self)
        xy_size_label=QLabel("XY:", self)
        z_size_label=QLabel("Z:", self)
        self.xy_size=QLineEdit(self, placeholderText='None')
        self.xy_size.setValidator(QDoubleValidator(bottom=0)) # non-negative floats only
        self.z_size=QLineEdit(self, placeholderText='None')
        self.z_size.setValidator(QDoubleValidator(bottom=0)) # non-negative floats only

        self.voxel_size_HLayout.addWidget(xy_size_label)
        self.voxel_size_HLayout.addWidget(self.xy_size)
        self.voxel_size_HLayout.addWidget(z_size_label)
        self.voxel_size_HLayout.addWidget(self.z_size)

        self.voxel_size_VLayout.addWidget(voxel_size_label)
        self.voxel_size_VLayout.addLayout(self.voxel_size_HLayout)

        # Tabbed Menu
        self.tabbed_menu_widget = QTabWidget()
        self.tabbed_menu_widget.addTab(self.get_segmentation_tab(), "Segmentation")
        self.tabbed_menu_widget.addTab(self.get_FUCCI_tab(), "FUCCI")
        self.tabbed_menu_widget.addTab(self.get_tracking_tab(), "Tracking")
        self.tabbed_menu_widget.addTab(self.get_volumes_tab(), 'Volumes')

        # Command Line Interface
        self.command_line_button=QPushButton("Open Command Line", self)

        # Save Menu
        save_widget=QWidget(objectName='bordered')
        save_menu=QGridLayout(save_widget)
        save_menu.setVerticalSpacing(5)
        self.save_button = QPushButton("Save", self)
        self.save_as_button = QPushButton("Save As", self)
        self.save_stack = QCheckBox("Save Stack", self)
        self.also_save_tracking=QCheckBox("Save Tracking", self)

        save_menu.addWidget(self.save_button, 0, 0)
        save_menu.addWidget(self.save_as_button, 0, 1)
        save_menu.addWidget(self.save_stack, 1, 0)
        save_menu.addWidget(self.also_save_tracking, 1, 1)
        
        left_toolbar = QWidget()
        left_toolbar_layout = QVBoxLayout(left_toolbar)

        left_scroll_area=QScrollArea()
        left_scroll_area.setWidgetResizable(True)
        left_scroll_area.setWidget(left_toolbar)
        left_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff) # disable horizontal scroll bar
        left_scroll_area.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred) # restore horizontal size policy
        left_toolbar_layout.setSpacing(10)
        left_toolbar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        left_toolbar_layout.addLayout(open_menu)
        left_toolbar_layout.addWidget(LUT_widget)
        left_toolbar_layout.addWidget(self.voxel_size_widget)
        left_toolbar_layout.addWidget(self.tabbed_menu_widget)
        left_toolbar_layout.addWidget(self.command_line_button)
        left_toolbar_layout.addWidget(save_widget)
        
        # normalize
        self.normalize_frame_button.toggled.connect(self.update_normalize_frame)
        self.normalize_stack_button.toggled.connect(self.update_normalize_frame)
        self.normalize_custom_button.toggled.connect(self.update_normalize_frame)
        # segmentation overlay
        self.masks_checkbox.stateChanged.connect(self.canvas.overlay_masks)
        self.outlines_checkbox.stateChanged.connect(self.canvas.overlay_outlines)
        # command line
        self.command_line_button.clicked.connect(self.open_command_line)
        # voxel size
        self.xy_size.editingFinished.connect(self.update_voxel_size)
        self.z_size.editingFinished.connect(self.update_voxel_size)
        # save
        self.save_button.clicked.connect(self.save_segmentation)
        self.save_as_button.clicked.connect(self.save_as_segmentation)

        # switch tabs
        self.tabbed_menu_widget.currentChanged.connect(self.tab_switched)

        return left_scroll_area
    
    def tab_switched(self, index):
        if not self.file_loaded:
            self.current_tab=index
            return
        
        # save visual settings for the previous tab
        self.saved_visual_settings[self.current_tab]=self.visual_settings

        # load visual settings for the new tab
        self.current_tab=index
        self.visual_settings=self.saved_visual_settings[index]

        self.highlight_track_ends()
        self.FUCCI_overlay()

    @property
    def visual_settings(self):
        # retrieve the current visual settings
        out={'RGB': self.get_RGB(),
             'normalize_type': self.normalize_type,
             'masks': self.masks_checkbox.isChecked(),
             'outlines': self.outlines_checkbox.isChecked(),
             'LUTs': self.LUT_slider_values
             }
        return out
    
    @visual_settings.setter
    def visual_settings(self, settings):
        if settings['RGB'] is not None: # RGB
            self.set_RGB(settings[0])
        self.normalize_type=settings['normalize_type']
        self.masks_checkbox.setChecked(settings['masks'])
        self.outlines_checkbox.setChecked(settings['outlines'])
        if settings['LUTs'] is not None: # LUT
            self.LUT_slider_values=settings['LUTs']

    def get_segmentation_tab(self):
        segmentation_tab=QWidget()
        segmentation_layout=QVBoxLayout(segmentation_tab)
        segmentation_layout.setSpacing(5)
        segmentation_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        segment_frame_widget=QWidget(objectName='bordered')
        segment_frame_layout=QVBoxLayout(segment_frame_widget)
        self.cell_diameter=QLineEdit(self, placeholderText='Auto')
        self.cell_diameter.setValidator(QDoubleValidator(bottom=0)) # non-negative floats only
        self.cell_diameter.setFixedWidth(60)
        self.cell_diameter_calibrate=QPushButton("Calibrate", self)
        self.cell_diameter_calibrate.setFixedWidth(70)
        self.cell_diameter_layout=QHBoxLayout()
        self.cell_diameter_layout.setSpacing(5)
        self.cell_diameter_layout.addWidget(QLabel("Cell Diameter:", self))
        self.cell_diameter_layout.addWidget(self.cell_diameter)
        self.cell_diameter_layout.addWidget(self.cell_diameter_calibrate)

        # channel selection
        self.segmentation_channels_widget=QWidget()
        self.segmentation_channels_widget.setContentsMargins(0, 0, 0, 0)
        self.segmentation_channels_layout=QVBoxLayout(self.segmentation_channels_widget)
        self.add_channel_layout(self.segmentation_channels_layout)

        segmentation_button_layout=QHBoxLayout()
        self.segment_frame_button=QPushButton("Segment Frame", self)
        self.segment_stack_button=QPushButton("Segment Stack", self)
        segmentation_button_layout.addWidget(self.segment_frame_button)
        segmentation_button_layout.addWidget(self.segment_stack_button)

        # segmentation utilities
        segmentation_utils_widget=QWidget(objectName='bordered')
        segmentation_utils_layout=QVBoxLayout(segmentation_utils_widget)
        operate_on_label=QLabel("Operate on:", self)
        operate_on_layout=QHBoxLayout()
        self.segment_on_frame=QRadioButton("Frame", self)
        self.segment_on_stack=QRadioButton("Stack", self)
        self.segment_on_frame.setChecked(True)
        mend_remove_layout=QHBoxLayout()
        self.mend_gaps_button=QPushButton("Mend Gaps", self)
        self.remove_edge_masks_button=QPushButton("Remove Edge Masks", self)
        mend_remove_layout.addWidget(self.mend_gaps_button)
        mend_remove_layout.addWidget(self.remove_edge_masks_button)
        self.ROIs_label=QLabel("0 ROIs", self)
        gap_size_layout=QHBoxLayout()
        gap_size_label=QLabel("Gap Size:", self)
        self.gap_size=QLineEdit(self, placeholderText='Auto')
        self.gap_size.setValidator(QIntValidator(bottom=0)) # non-negative integers only
        gap_size_layout.addWidget(gap_size_label)
        gap_size_layout.addWidget(self.gap_size)
        generate_remove_layout=QHBoxLayout()
        generate_outlines_button=QPushButton("Generate Outlines", self)
        clear_masks_button=QPushButton("Clear Masks", self)
        generate_remove_layout.addWidget(generate_outlines_button)
        generate_remove_layout.addWidget(clear_masks_button)
        operate_on_layout.addWidget(self.segment_on_frame)
        operate_on_layout.addWidget(self.segment_on_stack)
        segmentation_button_layout.addWidget(self.mend_gaps_button)
        segmentation_button_layout.addWidget(self.remove_edge_masks_button)

        segment_frame_layout.addLayout(self.cell_diameter_layout)
        segment_frame_layout.addWidget(self.segmentation_channels_widget)
        segment_frame_layout.addSpacerItem(self.vertical_spacer())
        segment_frame_layout.addWidget(self.ROIs_label)
        segment_frame_layout.addLayout(segmentation_button_layout)

        segmentation_utils_layout.addWidget(operate_on_label)
        segmentation_utils_layout.addLayout(operate_on_layout)
        segmentation_utils_layout.addLayout(mend_remove_layout)
        segmentation_utils_layout.addLayout(gap_size_layout)
        segmentation_utils_layout.addLayout(generate_remove_layout)

        segmentation_layout.addWidget(segment_frame_widget)
        segmentation_layout.addWidget(segmentation_utils_widget)

        self.mend_gaps_button.clicked.connect(self.mend_gaps)
        self.remove_edge_masks_button.clicked.connect(self.remove_edge_masks)
        self.cell_diameter.textChanged.connect(self.update_cell_diameter)
        self.cell_diameter_calibrate.clicked.connect(self.calibrate_diameter_pressed)
        self.segment_frame_button.clicked.connect(self.segment_frame_pressed)
        self.segment_stack_button.clicked.connect(self.segment_stack_pressed)
        generate_outlines_button.clicked.connect(self.generate_outlines_list)
        clear_masks_button.clicked.connect(self.clear_masks)
        self.circle_mask=None

        return segmentation_tab
    
    def update_ROIs_label(self):
        if not self.file_loaded:
            return
        
        self.ROIs_label.setText(f'{self.frame.n_cells} ROIs')

    def cell_prompt_changed(self, cell_n):
        if not self.file_loaded:
            return
        
        if cell_n=='' or cell_n=='None':
            cell_n=None
            self.update_tracking_ID_label(None)
            return
        else:
            cell_n=int(cell_n)

        self.select_cell(cell=cell_n)

    def particle_prompt_changed(self, particle):
        if not self.file_loaded:
            return
        
        if particle=='' or particle=='None':
            particle=None
            self.update_cell_label(None)
            return
        else:
            particle=int(particle)

        self.select_cell(particle=particle)

    def mend_gaps(self):
        if not self.file_loaded:
            return
        from segmentation_tools.preprocessing import mend_gaps
        if self.segment_on_stack.isChecked():
            frames=self.stack.frames
        else:
            frames=[self.frame]
        
        gap_size=self.gap_size.text()
        for frame in self.progress_bar(frames):
            if gap_size=='':
                gap_size=frame.mean_cell_area(scaled=False)/2 # default to half the mean cell area
            else:
                gap_size=int(gap_size)
            new_masks, mended=mend_gaps(frame.masks, gap_size)

            if mended:
                changed_cells=np.unique(new_masks[new_masks!=frame.masks])
                changed_masks=np.zeros_like(frame.masks, dtype=int)
                changed_masks_bool=np.isin(new_masks, changed_cells)
                changed_masks[changed_masks_bool]=new_masks[changed_masks_bool]
                if frame.has_outlines:
                    outlines_list=utils.outlines_list(changed_masks)
                    for cell, o in zip(frame.cells[changed_cells], outlines_list):
                        cell.outline=o
                        cell.get_centroid()

                frame.masks=new_masks
                frame.outlines=utils.masks_to_outlines(frame.masks)
                if hasattr(frame, 'stored_mask_overlay'):
                    del frame.stored_mask_overlay
        
        self.update_display()

    def remove_edge_masks(self):
        if not self.file_loaded:
            return
        if self.segment_on_stack.isChecked():
            frames=self.stack.frames
        else:
            frames=[self.frame]
        
        for frame in frames:
            top=frame.masks[0]
            bottom=frame.masks[-1]
            left=frame.masks[1:-1,0]
            right=frame.masks[1:-1,-1]
            edge_cells=np.unique(np.concatenate([top, bottom, left, right]))
            edge_cells=edge_cells[edge_cells!=0] # remove background

            edge_cells-=1 # convert to 0-indexed
            if len(edge_cells)>0:
                idx=frame.delete_cells(edge_cells)
                self.remove_tracking_data(edge_cells, idx, frame_number=frame.frame_number)
                
                #frame.masks[changed_masks_bool]=0
                frame.outlines=utils.masks_to_outlines(frame.masks)

                print(f'Removed {len(edge_cells)} edge masks')
                if hasattr(frame, 'stored_mask_overlay'):
                    del frame.stored_mask_overlay
                
                if frame==self.frame:
                    if self.selected_cell_n in edge_cells:
                        # deselect the removed cell if it was selected
                        self.select_cell(None)
                    self.canvas.draw_outlines()
                    self.highlight_track_ends()
                    self.update_display()
                    self.update_ROIs_label()
    
    def update_cell_diameter(self, diameter):
        self.draw_cell_diameter(diameter)
    
    def draw_cell_diameter(self, diameter):
        if self.circle_mask is not None:
            self.canvas.img_plot.removeItem(self.circle_mask)

        if diameter=='':
            return
        
        diameter=float(diameter)
        padding=5
        img_shape=self.canvas.img_data.shape[:2]
        self.circle_mask=QGraphicsEllipseItem(padding, img_shape[1]+padding, diameter, diameter)
        self.circle_mask.setBrush(pg.mkBrush(color='#4A90E2'))
        self.canvas.img_plot.addItem(self.circle_mask)

    def calibrate_diameter_pressed(self):
        channels=[self.membrane_channel.currentIndex(), self.nuclear_channel.currentIndex()]
        diam=self.calibrate_cell_diameter(self.frame.img, channels)

        print(f'Computed cell diameter {diam:.2f} with channels {channels}')
        self.cell_diameter.setText(f'{diam:.2f}')

    def calibrate_cell_diameter(self, img, channels):
        if not self.file_loaded:
            return

        if not hasattr(self, 'size_model'):
            from cellpose import models
            model_type='cyto3'
            self.cellpose_model=models.CellposeModel(gpu=True, model_type=model_type)
            self.size_model_path=models.size_model_path(model_type)
            self.size_model=models.SizeModel(self.cellpose_model, pretrained_size=self.size_model_path)

        if channels[1]==4: # FUCCI channel
            from segmentation_tools.image_segmentation import combine_FUCCI_channels
            if channels[0]==0:
                membrane=np.mean(img, axis=-1) # grayscale
            else:
                membrane=img[...,channels[0]-1] # fetch specified channel

            nuclei=combine_FUCCI_channels(img)[..., 0]
            img=np.stack([nuclei, membrane], axis=-1)
            diam, style_diam=self.size_model.eval(img, channels=[2,1])
        else:
            diam, style_diam=self.size_model.eval(img, channels=channels)

        return diam

    def segment_frame_pressed(self):
        if not self.file_loaded:
            return
        
        self.segment([self.frame])

        # update the display
        self.cell_diameter.setText(f'{self.frame.cell_diameter:.2f}')
        self.masks_checkbox.setChecked(True)
        self.canvas.draw_masks()
        self.update_display()
        self.FUCCI_overlay()

    def segment_stack_pressed(self):
        if not self.file_loaded:
            return
        
        if self.is_zstack: # segment the zstack
            for frame in self.stack.frames:
                frame.img=frame.zstack[self.zstack_number]

        self.segment(self.stack.frames)

        # update the display
        self.cell_diameter.setText(f'{self.frame.cell_diameter:.2f}')
        self.masks_checkbox.setChecked(True)
        self.canvas.draw_masks()
        self.update_display()
        self.FUCCI_overlay()

    def segment(self, frames):
        diameter=self.cell_diameter.text()
        if diameter=='':
            diameter=None
        else:
            diameter=float(diameter)
        channels=[self.membrane_channel.currentIndex(), self.nuclear_channel.currentIndex()]

        if not hasattr(self, 'cellpose_model'):
            from cellpose import models
            model_type='cyto3'
            self.cellpose_model=models.CellposeModel(gpu=True, model_type=model_type)
        
        for frame in self.progress_bar(frames, desc='Segmenting frames'):
            if diameter is None:
                diameter=self.calibrate_cell_diameter(frame.img, channels)
            frame.cell_diameter=diameter
            if channels[1]==4: # FUCCI channel
                img=frame.img.copy()
                from segmentation_tools.image_segmentation import combine_FUCCI_channels
                if channels[0]==0:
                    membrane=np.mean(img, axis=-1) # grayscale
                else:
                    membrane=img[...,channels[0]-1] # fetch specified channel

                nuclei=combine_FUCCI_channels(img)[..., 0]
                img=np.stack([nuclei, membrane], axis=-1)
                masks, _, _=self.cellpose_model.eval(img, channels=[2,1], diameter=diameter)
            else:
                masks, _, _=self.cellpose_model.eval(frame.img, channels=channels, diameter=diameter)
            frame.masks=masks
            self.replace_segmentation(frame)

            if frame==self.frame:
                self.update_ROIs_label()

    def clear_masks(self):
        if not self.file_loaded:
            return
        
        if self.segment_on_stack.isChecked():
            frames=self.stack.frames
        else:
            frames=[self.frame]
        
        for frame in frames:
            frame.masks=np.zeros_like(frame.masks)
            self.replace_segmentation(frame)
            if hasattr(self.stack, 'tracked_centroids'):
                t=self.stack.tracked_centroids
                self.stack.tracked_centroids=t[t.frame!=frame.frame_number]
                self.also_save_tracking.setChecked(True)

        self.update_display()

    def progress_bar(self, iterable, desc=None, length=None):
        if length is None:
            length=len(iterable)

        if length == 1:
            return iterable
        else:
            # Initialize tqdm progress bar
            tqdm_bar = tqdm(iterable, desc=desc, total=length)
            
            # Initialize QProgressBar
            qprogress_bar = QProgressBar()
            qprogress_bar.setMaximum(length)

            # Set size policy to match the status bar width
            qprogress_bar.setFixedHeight(int(self.statusBar().height()*0.8))
            qprogress_bar.setFixedWidth(int(self.statusBar().width()*0.2))

            # Temporarily hide existing permanent status bar widgets
            self.status_coordinates.setVisible(False)
            self.status_pixel_value.setVisible(False)

            self.statusBar().addPermanentWidget(qprogress_bar)

            # Custom iterator to update both progress bars
            def custom_iterator():
                self.is_iterating=True
                self.cancel_iter=False
                for i, item in enumerate(iterable):
                    if self.cancel_iter:
                        print('Task cancelled.')
                        break
                    yield item
                    tqdm_bar.update(1)
                    qprogress_bar.setValue(i + 1)
                tqdm_bar.close()
                self.statusBar().removeWidget(qprogress_bar)
                # Restore existing permanent status bar widgets
                self.status_coordinates.setVisible(True)
                self.status_pixel_value.setVisible(True)
                self.is_iterating=False

            return custom_iterator()
        
    def replace_segmentation(self, frame):
        ''' Generate the cell outlines and cell objects from the new segmentation masks. '''
        frame.has_outlines=False
        frame.outlines=utils.masks_to_outlines(frame.masks)
        frame.n_cells=np.max(frame.masks)
        frame.cells = np.array([Cell(n, np.empty((0,2)), frame_number=frame.frame_number, parent=frame) for n in range(frame.n_cells)])
        if hasattr(frame, 'stored_mask_overlay'):
            del frame.stored_mask_overlay

    def get_FUCCI_tab(self):
        FUCCI_tab = QWidget()
        FUCCI_tab_layout = QVBoxLayout(FUCCI_tab)
        FUCCI_tab_layout.setSpacing(10)
        FUCCI_tab_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        get_intensities_button=QPushButton("Get Cell Red/Green Intensities", self)
        get_tracked_FUCCI_button=QPushButton("FUCCI from tracks", self)
        measure_FUCCI_widget=QWidget(objectName='bordered')
        measure_FUCCI_layout=QVBoxLayout(measure_FUCCI_widget)

        red_threshold_layout=QHBoxLayout()
        red_threshold_label=QLabel("Red Threshold:", self)
        self.red_threshold=QLineEdit(self, placeholderText='Auto')
        self.red_threshold.setFixedWidth(60)
        self.red_threshold.setValidator(QDoubleValidator(bottom=0)) # non-negative floats only
        red_threshold_layout.addWidget(red_threshold_label)
        red_threshold_layout.addWidget(self.red_threshold)

        green_threshold_layout=QHBoxLayout()
        green_threshold_label=QLabel("Green Threshold:", self)
        self.green_threshold=QLineEdit(self, placeholderText='Auto')
        self.green_threshold.setFixedWidth(60)
        self.green_threshold.setValidator(QDoubleValidator(bottom=0)) # non-negative floats only
        green_threshold_layout.addWidget(green_threshold_label)
        green_threshold_layout.addWidget(self.green_threshold)

        percent_threshold_layout=QHBoxLayout()
        percent_threshold_label=QLabel("Minimum N/C Ratio:", self)
        self.percent_threshold=QLineEdit(self, placeholderText='0.15')
        self.percent_threshold.setFixedWidth(60)
        self.percent_threshold.setValidator(QDoubleValidator(bottom=0)) # non-negative floats only
        percent_threshold_layout.addWidget(percent_threshold_label)
        percent_threshold_layout.addWidget(self.percent_threshold)

        FUCCI_button_layout=QHBoxLayout()
        FUCCI_button_layout.setSpacing(5)
        FUCCI_frame_button=QPushButton("Measure Frame", self)
        FUCCI_stack_button=QPushButton("Measure Stack", self)
        FUCCI_button_layout.addWidget(FUCCI_frame_button)
        FUCCI_button_layout.addWidget(FUCCI_stack_button)

        annotate_FUCCI_widget=QWidget(objectName='bordered')
        annotate_FUCCI_layout=QVBoxLayout(annotate_FUCCI_widget)
        FUCCI_overlay_layout=QHBoxLayout()
        overlay_label = QLabel("FUCCI overlay: ", self)
        self.FUCCI_dropdown = QComboBox(self)
        self.FUCCI_dropdown.addItems(["None", "Green", "Red", "All"])
        FUCCI_overlay_layout.addWidget(overlay_label)
        FUCCI_overlay_layout.addWidget(self.FUCCI_dropdown)
        self.FUCCI_checkbox = QCheckBox("Show FUCCI Channel", self)
        # clear FUCCI, propagate FUCCI
        self.propagate_FUCCI_checkbox=QCheckBox("Propagate FUCCI", self)
        self.propagate_FUCCI_checkbox.setEnabled(False)
        clear_frame_button=QPushButton("Clear Frame", self)
        clear_stack_button=QPushButton("Clear Stack", self)
        clear_FUCCI_layout=QHBoxLayout()
        clear_FUCCI_layout.addWidget(clear_frame_button)
        clear_FUCCI_layout.addWidget(clear_stack_button)

        measure_FUCCI_layout.addLayout(red_threshold_layout)
        measure_FUCCI_layout.addLayout(green_threshold_layout)
        measure_FUCCI_layout.addLayout(percent_threshold_layout)
        measure_FUCCI_layout.addSpacerItem(self.vertical_spacer())
        measure_FUCCI_layout.addLayout(FUCCI_button_layout)

        annotate_FUCCI_layout.addLayout(FUCCI_overlay_layout)
        annotate_FUCCI_layout.addWidget(self.FUCCI_checkbox)
        annotate_FUCCI_layout.addSpacerItem(self.vertical_spacer())
        annotate_FUCCI_layout.addWidget(self.propagate_FUCCI_checkbox)
        annotate_FUCCI_layout.addLayout(clear_FUCCI_layout)

        FUCCI_tab_layout.addWidget(get_intensities_button)
        FUCCI_tab_layout.addWidget(get_tracked_FUCCI_button)
        FUCCI_tab_layout.addWidget(measure_FUCCI_widget)
        FUCCI_tab_layout.addWidget(annotate_FUCCI_widget)

        get_intensities_button.clicked.connect(self.cell_red_green_intensities)
        get_tracked_FUCCI_button.clicked.connect(self.get_tracked_FUCCI)
        self.FUCCI_dropdown.currentIndexChanged.connect(self.FUCCI_overlay_changed)
        self.FUCCI_checkbox.stateChanged.connect(self.update_display)
        FUCCI_frame_button.clicked.connect(self.measure_FUCCI_frame)
        FUCCI_stack_button.clicked.connect(self.measure_FUCCI_stack)
        self.propagate_FUCCI_checkbox.stateChanged.connect(self.propagate_FUCCI_toggled)
        clear_frame_button.clicked.connect(self.clear_FUCCI_frame_pressed)
        clear_stack_button.clicked.connect(self.clear_FUCCI_stack_pressed)

        return FUCCI_tab
    
    def get_tracked_FUCCI(self):
        if not self.file_loaded:
            return
        if not hasattr(self.stack, 'tracked_centroids'):
            self.statusBar().showMessage('No tracked centroids found.', 2000)
            return
        
        self.stack.measure_FUCCI_by_transitions(progress=self.progress_bar)

        for frame in self.stack.frames:
            self.get_red_green(frame)

        self.FUCCI_overlay()

    def cell_red_green_intensities(self, event=None, percentile=90, sigma=4):
        if not self.file_loaded:
            return
        for frame in self.progress_bar(self.stack.frames):
            frame.get_red_green_intensities(percentile, sigma)

    def propagate_FUCCI_toggled(self, state):
        ''' Propagate the FUCCI labels forward in time. '''
        if state!=2 or not self.file_loaded:
            return
        if not hasattr(self.stack, 'tracked_centroids'):
            self.statusBar().showMessage('No tracked centroids found.', 2000)
            return
        
        self.convert_red_green()
        self.stack.propagate_FUCCI_labels()
        for frame in self.stack.frames:
            self.get_red_green(frame)
        self.FUCCI_overlay()
    
    def clear_FUCCI_frame_pressed(self):
        if not self.file_loaded:
            return
        self.clear_FUCCI([self.frame])

    def clear_FUCCI_stack_pressed(self):
        if not self.file_loaded:
            return
        self.clear_FUCCI(self.stack.frames)

    def clear_FUCCI(self, frames):
        for frame in frames:
            frame.set_cell_attrs(['red', 'green'], np.array([[False, False] for _ in range(frame.n_cells)]).T)
        self.FUCCI_overlay()

    def vertical_spacer(self, spacing=None, hSizePolicy=QSizePolicy.Policy.Fixed, vSizePolicy=QSizePolicy.Policy.Fixed):
        if spacing is None:
            spacing=self.spacer
        return QSpacerItem(*spacing, hSizePolicy, vSizePolicy)
    
    def measure_FUCCI_frame(self):
        if not self.file_loaded:
            return
        self.measure_FUCCI([self.frame])

    def measure_FUCCI_stack(self):
        if not self.file_loaded:
            return
        self.measure_FUCCI(self.stack.frames)

    def get_FUCCI_thresholds(self):
        red_threshold=self.red_threshold.text()
        green_threshold=self.green_threshold.text()
        percent_threshold=self.percent_threshold.text()

        if red_threshold=='':
            red_threshold=None
        else:
            red_threshold=float(red_threshold)
        if green_threshold=='':
            green_threshold=None
        else:
            green_threshold=float(green_threshold)
        if percent_threshold=='':
            percent_threshold=0.15
        else:
            percent_threshold=float(percent_threshold)
        return red_threshold, green_threshold, percent_threshold
    
    def measure_FUCCI(self, frames):
        red_threshold, green_threshold, percent_threshold=self.get_FUCCI_thresholds()
        for frame in self.progress_bar(frames, desc='Measuring FUCCI'):
            if self.is_zstack:
                img=frame.zstack[self.zstack_number]
            else:
                img=frame.img
            if not hasattr(frame, 'FUCCI'):
                frame.FUCCI=img[...,0], img[...,1] # use the red and green channels

            frame.measure_FUCCI(red_fluor_threshold=red_threshold, green_fluor_threshold=green_threshold, orange_brightness=1, percent_threshold=percent_threshold)
            self.get_red_green(frame)
        
        red_threshold=self.frame.red_fluor_threshold
        green_threshold=self.frame.green_fluor_threshold
        self.red_threshold.setText(f'{red_threshold:.2f}')
        self.green_threshold.setText(f'{green_threshold:.2f}')
        self.FUCCI_dropdown.setCurrentIndex(3)
        self.FUCCI_overlay()

    def get_volumes_tab(self):
        self.volumes_tab=QWidget()
        volumes_layout=QVBoxLayout(self.volumes_tab)
        volumes_layout.setSpacing(10)
        volumes_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        operate_on_label=QLabel("Operate on:", self)
        operate_on_layout=QHBoxLayout()
        self.volumes_on_frame=QRadioButton("Frame", self)
        self.volumes_on_stack=QRadioButton("Stack", self)
        operate_on_layout.addWidget(self.volumes_on_frame)
        operate_on_layout.addWidget(self.volumes_on_stack)
        self.volumes_on_frame.setChecked(True)
        self.get_heights_layout=QHBoxLayout()
        self.get_heights_button=QPushButton("Measure Heights", self)
        self.volume_button=QPushButton("Measure Volumes", self)
        self.get_heights_layout.addWidget(self.get_heights_button)
        self.get_heights_layout.addWidget(self.volume_button)
        peak_prominence_label=QLabel("Peak Prominence (0 to 1):", self)
        self.peak_prominence=QLineEdit(self, text='0.01', placeholderText='0.01')
        self.peak_prominence.setValidator(QDoubleValidator(bottom=0)) # non-negative floats only
        self.peak_prominence.setFixedWidth(60)
        self.peak_prominence_layout=QHBoxLayout()
        self.peak_prominence_layout.addWidget(peak_prominence_label)
        self.peak_prominence_layout.addWidget(self.peak_prominence)
        self.get_coverslip_height_layout=QHBoxLayout()
        coverslip_height_label=QLabel("Coverslip Height (μm):", self)
        self.coverslip_height=QLineEdit(self, placeholderText='Auto')
        self.coverslip_height.setValidator(QDoubleValidator(bottom=0)) # non-negative floats only
        self.coverslip_height.setFixedWidth(60)
        self.get_coverslip_height=QPushButton("Calibrate", self)
        self.get_coverslip_height_layout.addWidget(coverslip_height_label)
        self.get_coverslip_height_layout.addWidget(self.coverslip_height)
        self.get_coverslip_height_layout.addWidget(self.get_coverslip_height)
        self.get_spherical_volumes=QPushButton("Compute Spherical Volumes", self)

        self.volume_button.clicked.connect(self.measure_volumes)
        self.get_heights_button.clicked.connect(self.measure_heights)
        self.get_coverslip_height.clicked.connect(self.calibrate_coverslip_height)
        self.get_spherical_volumes.clicked.connect(self.compute_spherical_volumes)

        volumes_layout.addWidget(operate_on_label)
        volumes_layout.addLayout(operate_on_layout)
        volumes_layout.addLayout(self.get_heights_layout)
        volumes_layout.addLayout(self.peak_prominence_layout)
        volumes_layout.addLayout(self.get_coverslip_height_layout)
        volumes_layout.addWidget(self.get_spherical_volumes)

        return self.volumes_tab
    
    def get_tracking_tab(self):
        tracking_tab = QWidget()
        tracking_tab_layout = QVBoxLayout(tracking_tab)
        tracking_tab_layout.setSpacing(10)
        tracking_tab_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.tracking_range_layout=QFormLayout()
        range_label = QLabel('Search Range:  ', self)
        memory_label = QLabel('Memory:  ', self)
        self.memory_range = QLineEdit(self, placeholderText='0')
        self.tracking_range = QLineEdit(self, placeholderText='Auto')
        self.tracking_range.setValidator(QDoubleValidator(bottom=0)) # non-negative floats only
        self.memory_range.setValidator(QIntValidator(bottom=0)) # non-negative integers only
        self.tracking_range_layout.addRow(range_label, self.tracking_range)
        self.tracking_range_layout.addRow(memory_label, self.memory_range)

        self.track_centroids_button = QPushButton("Track Centroids", self)

        io_menu=QHBoxLayout()
        self.save_tracking_button=QPushButton("Save Tracking", self)
        self.load_tracking_button=QPushButton("Load Tracking", self)
        io_menu.addWidget(self.save_tracking_button)
        io_menu.addWidget(self.load_tracking_button)

        self.highlight_track_ends_checkbox=QCheckBox("Highlight Track Ends", self)
        split_particle_button=QPushButton("Split Particle", self)
        delete_particle_label=QLabel("Delete Particle:", self)
        delete_particle_layout=QHBoxLayout()
        delete_head=QPushButton("Head", self)
        delete_tail=QPushButton("Tail", self)
        delete_all=QPushButton("All", self)
        delete_particle_layout.addWidget(delete_head)
        delete_particle_layout.addWidget(delete_tail)
        delete_particle_layout.addWidget(delete_all)
        clear_tracking_button=QPushButton("Clear Tracking", self)

        track_centroids_widget=QWidget(objectName='bordered')
        track_centroids_layout=QVBoxLayout(track_centroids_widget)
        edit_tracking_widget=QWidget(objectName='bordered')
        edit_tracking_layout=QVBoxLayout(edit_tracking_widget)
        track_centroids_layout.addLayout(self.tracking_range_layout)
        track_centroids_layout.addWidget(self.track_centroids_button)
        edit_tracking_layout.addWidget(self.highlight_track_ends_checkbox)
        edit_tracking_layout.addWidget(split_particle_button)
        edit_tracking_layout.addWidget(delete_particle_label)
        edit_tracking_layout.addLayout(delete_particle_layout)
        edit_tracking_layout.addWidget(clear_tracking_button)
        edit_tracking_layout.addSpacerItem(self.vertical_spacer())
        edit_tracking_layout.addLayout(io_menu)
        
        tracking_tab_layout.addWidget(track_centroids_widget)
        tracking_tab_layout.addWidget(edit_tracking_widget)

        self.track_centroids_button.clicked.connect(self.track_centroids)
        self.tracking_range.returnPressed.connect(self.track_centroids)
        split_particle_button.clicked.connect(self.split_particle_tracks)
        clear_tracking_button.clicked.connect(self.clear_tracking)
        self.save_tracking_button.clicked.connect(self.save_tracking)
        self.load_tracking_button.clicked.connect(self.load_tracking_pressed)
        self.highlight_track_ends_checkbox.stateChanged.connect(self.highlight_track_ends)
        delete_head.clicked.connect(self.delete_particle_head)
        delete_tail.clicked.connect(self.delete_particle_tail)
        delete_all.clicked.connect(self.delete_particle)

        return tracking_tab

    def delete_particle_head(self):
        if not self.file_loaded:
            return
        if not hasattr(self.stack, 'tracked_centroids'):
            self.delete_cell_mask(self.selected_cell_n)
            return
        
        else:
            particle_n=self.selected_particle_n
            current_frame_n=self.frame_number
            t=self.stack.tracked_centroids

            head_cell_numbers, head_frame_numbers=np.array(t[(t.particle==particle_n)&(t.frame<=current_frame_n)][['cell_number', 'frame']]).T
            self.also_save_tracking.setChecked(True)
            for cell_n, frame_n in zip(head_cell_numbers, head_frame_numbers):
                frame=self.stack.frames[frame_n]
                if hasattr(frame, 'stored_mask_overlay'):
                    self.canvas.add_cell_highlight(cell_n, frame, color='none', layer='mask')
                self.delete_cell_mask(cell_n, frame)

            # reselect the particle
            self.selected_particle_n=particle_n
            self.plot_particle_statistic()

    def delete_particle_tail(self):
        if not self.file_loaded:
            return
        if not hasattr(self.stack, 'tracked_centroids'):
            self.delete_cell_mask(self.selected_cell_n)
            return
        
        else:
            particle_n=self.selected_particle_n
            current_frame_n=self.frame_number
            t=self.stack.tracked_centroids

            head_cell_numbers, head_frame_numbers=np.array(t[(t.particle==particle_n)&(t.frame>=current_frame_n)][['cell_number', 'frame']]).T
            self.also_save_tracking.setChecked(True)
            for cell_n, frame_n in zip(head_cell_numbers, head_frame_numbers):
                frame=self.stack.frames[frame_n]
                if hasattr(frame, 'stored_mask_overlay'):
                    self.canvas.add_cell_highlight(cell_n, frame, color='none', layer='mask')
                self.delete_cell_mask(cell_n, frame)

            # reselect the particle
            self.selected_particle_n=particle_n
            self.plot_particle_statistic()

    def delete_particle(self):
        if not self.file_loaded:
            return
        if not hasattr(self.stack, 'tracked_centroids'):
            self.delete_cell_mask(self.selected_cell_n)
            return
        
        else:
            particle_n=self.selected_particle_n
            t=self.stack.tracked_centroids
            self.also_save_tracking.setChecked(True)

            head_cell_numbers, head_frame_numbers=np.array(t[t.particle==particle_n][['cell_number', 'frame']]).T
            for cell_n, frame_n in zip(head_cell_numbers, head_frame_numbers):
                frame=self.stack.frames[frame_n]
                if hasattr(frame, 'stored_mask_overlay'):
                    self.canvas.add_cell_highlight(cell_n, frame, color='none', layer='mask')
                self.delete_cell_mask(cell_n, frame)

    def clear_tracking(self):
        if not self.file_loaded:
            return
        if not hasattr(self.stack, 'tracked_centroids'):
            return
        
        del self.stack.tracked_centroids
        self.canvas.clear_tracking_overlay()
        self.clear_particle_statistic()
        self.random_recolor() # recolor masks to signify unlinking
        
    def random_recolor(self):
        if not self.file_loaded:
            return
        for frame in self.stack.frames:
            if hasattr(frame, 'stored_mask_overlay'):
                del frame.stored_mask_overlay
            for cell in frame.cells:
                del cell.color_ID
            
        self.canvas.draw_masks()

    def highlight_track_ends(self):
        if not self.file_loaded or self.tabbed_menu_widget.currentIndex()!=2:
            self.canvas.clear_tracking_overlay()
            return
        
        if self.highlight_track_ends_checkbox.isChecked() and hasattr(self.stack, 'tracked_centroids'):
            # get the start and end points of each track
            t=self.stack.tracked_centroids
            track_ends=t.groupby('particle').agg({'frame': ['first', 'last']})
            track_ends.columns=['start', 'end']
            track_ends=track_ends.reset_index()

            births=track_ends[track_ends.start==self.frame_number]['particle']
            deaths=track_ends[track_ends.end==self.frame_number]['particle']
            if self.frame_number==0:
                births=[]
            elif self.frame_number==len(self.stack.frames)-1:
                deaths=[]
            
            birth_cells=t[(t.frame==self.frame_number)&t.particle.isin(births)]['cell_number']
            death_cells=t[(t.frame==self.frame_number)&t.particle.isin(deaths)]['cell_number']
            both=np.intersect1d(birth_cells, death_cells)
            colors=['lime']*len(birth_cells)+['red']*len(death_cells)+['orange']*len(both) # TODO: pick better colors which don't overlap with FUCCI
            self.canvas.highlight_cells(np.concatenate([birth_cells, death_cells, both]), alpha=0.5, cell_colors=colors, layer='tracking', img_type='outlines')

        else:
            self.canvas.clear_tracking_overlay()

    def canvas_wheelEvent(self, event):
        if not self.file_loaded:
            return
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier: # shift+scroll = z-stack
            if self.is_zstack:
                if event.angleDelta().y() > 0: # scroll up = higher in z-stack
                    self.zstack_slider.setValue(min(self.zstack_slider.value() + 1, len(self.frame.zstack) - 1))
                else:
                    self.zstack_slider.setValue(max(self.zstack_slider.value() - 1, 0))
        else: # scroll = frame
            if event.angleDelta().y() < 0: # scroll down = next frame
                self.change_current_frame(min(self.frame_number + 1, len(self.stack.frames) - 1))
            else: # scroll up = previous frame
                self.change_current_frame(max(self.frame_number - 1, 0))

    def update_zstack_number(self, zstack_number):
        if not self.file_loaded:
            return
        self.zstack_number = zstack_number
        self.frame.img=self.frame.zstack[self.zstack_number]
        self.update_coordinate_label()
        self.imshow()
        self.normalize()

    def refresh_right_toolbar(self, cell_attr=None):
        if cell_attr is None:
            self.plot_histogram()
            self.show_seg_overlay()
            self.plot_particle_statistic()
        else:
            if self.histogram_menu.currentText()==cell_attr:
                self.plot_histogram()
            if self.seg_overlay_attr.currentText()==cell_attr:
                self.show_seg_overlay()
            if self.particle_stat_menu.currentText()==cell_attr:
                self.plot_particle_statistic()

    def measure_volumes(self):
        if not self.file_loaded:
            return
        if self.volumes_on_stack.isChecked():
            frames=self.stack.frames
        else:
            frames=[self.frame]

        for frame in self.progress_bar(frames):
            self.measure_frame_volumes(frame)

        # update the display if necessary
        self.refresh_right_toolbar('volume')

    def measure_frame_volumes(self, frame):
        if not hasattr(frame, 'heights'):
            if hasattr(frame, 'zstack'):
                self.measure_heights()
            else:
                raise ValueError(f'No heights or z-stack available to measure volumes for {frame.name}.')
        
        if not hasattr(frame, 'z_scale'):
            print(f'No z-scale available for {frame.name}. Defaulting to 1.')
            self.z_size.setText('1.0')
            self.update_voxel_size()
        if not hasattr(frame, 'scale'):
            print(f'No scale available for {frame.name}. Defaulting to 0.1625.')
            self.xy_size.setText('0.1625')
            self.update_voxel_size()
            frame.scale=0.1625 # 40x objective with 0.325 µm/pixel camera
        frame.get_volumes()
        return frame.volumes

    def new_histogram(self):
        self.plot_histogram()
        self.histogram.autoRange()

    def plot_histogram(self):
        self.histogram.clear()
        hist_attr=self.histogram_menu.currentText()
        self.histogram.setLabel('bottom', hist_attr)
        if hist_attr=='Select Cell Attribute':
            return
        # get the attribute values
        # TODO: check whether to operate on stack or frame
        cell_attrs=np.array(self.frame.get_cell_attrs(hist_attr, fill_value=np.nan))
        
        if np.all(np.isnan(cell_attrs)):
            return
        
        hist_data=np.array(cell_attrs)[~np.isnan(cell_attrs)]

        iqr=np.percentile(hist_data, 75)-np.percentile(hist_data, 25)
        bin_width=2*iqr/(len(hist_data)**(1/3))
        bins=np.arange(np.min(hist_data), np.max(hist_data)+bin_width, bin_width)
        
        n, bins=np.histogram(hist_data, bins=bins, density=True)
        self.histogram.plot(bins, n, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))

    def calibrate_coverslip_height(self):
        from segmentation_tools.heightmap import get_coverslip_z
        if not self.file_loaded:
            return
        if self.volumes_on_stack.isChecked():
            frames=self.stack.frames
            if not all(hasattr(frame, 'zstack') for frame in frames):
                raise ValueError('No z-stacks available to calibrate coverslip height.')
            if len(np.unique([frame.zstack.shape[0] for frame in frames]))>1:
                raise ValueError('Z-stack lengths are not consistent.')
        else:
            frames=[self.frame]

        z_profile=[]
        for z_index in range(frames[0].zstack.shape[0]):
            if self.is_grayscale:
                z_profile.append(np.mean(np.concatenate([frame.zstack[z_index].flatten() for frame in frames])))
            else:
                z_profile.append(np.mean(np.concatenate([frame.zstack[z_index,...,2].flatten() for frame in frames])))
        
        if not hasattr(self.frame, 'z_scale'):
            print(f'No z-scale available for {self.frame.name}. Defaulting to 1.')
            self.z_size.setText('1.0')
            self.update_voxel_size()
        scale=self.frame.z_scale

        coverslip_height=get_coverslip_z(z_profile, scale=scale, precision=0.01)
        for frame in frames:
            frame.coverslip_height=coverslip_height
        self.coverslip_height.setText(f'{coverslip_height:.2f}')

    def measure_heights(self):
        if not self.file_loaded:
            return
        from segmentation_tools.heightmap import get_heights
        if self.volumes_on_stack.isChecked():
            frames=self.stack.frames
        else:
            frames=[self.frame]
        
        peak_prominence=self.peak_prominence.text()
        if peak_prominence=='':
            peak_prominence=0.01
        else:
            peak_prominence=float(peak_prominence)

        coverslip_height=self.coverslip_height.text()
        if coverslip_height=='':
            self.calibrate_coverslip_height()
            coverslip_height=self.coverslip_height.text()
        coverslip_height=float(coverslip_height)
        
        for frame in self.progress_bar(frames):
            if not hasattr(frame, 'zstack'):
                raise ValueError(f'No z-stack available to measure heights for {frame.name}.')
            else:
                if self.is_grayscale:
                    membrane=frame.zstack
                else:
                    membrane=frame.zstack[..., 2] # TODO: allow user to specify membrane channel
                frame.heights=get_heights(membrane, peak_prominence=peak_prominence)
                frame.to_heightmap()
                frame.coverslip_height=coverslip_height
                self.show_seg_overlay()

        self.volume_button.setEnabled(True)

    def compute_spherical_volumes(self):
        if not self.file_loaded:
            return
        
        if self.volumes_on_stack.isChecked():
            frames=self.stack.frames
        else:
            frames=[self.frame]

        for frame in self.progress_bar(frames):
            frame.get_spherical_volumes()
        
        self.refresh_right_toolbar('volume')

    def change_current_frame(self, frame_number, reset=False):
        if not self.file_loaded:
            return
        self.frame_number = frame_number
        self.frame_slider.setValue(frame_number)
        self.frame=self.stack.frames[self.frame_number]
        self.globals_dict['frame']=self.frame

        if hasattr(self.frame, 'zstack'):
            self.frame.img=self.frame.zstack[self.zstack_number]
            self.zstack_slider.setVisible(True)
            self.zstack_slider.setRange(0, self.frame.zstack.shape[0]-1)
            self.is_zstack=True
        else:
            self.frame.img=self.frame.img
            self.zstack_slider.setVisible(False)
            self.is_zstack=False

        if self.is_zstack or hasattr(self.frame, 'heights'):
            self.volume_button.setEnabled(True)
            if not self.is_zstack: # enable/disable z-stack specific options
                self.get_heights_button.setEnabled(False)
                self.peak_prominence.setEnabled(False)
            else:
                self.get_heights_button.setEnabled(True)
                self.peak_prominence.setEnabled(True)

            if hasattr(self.frame, 'coverslip_height'):
                self.coverslip_height.setText(f'{self.frame.coverslip_height:.2f}')
            else:
                self.coverslip_height.setText('')
        else:
            self.get_heights_button.setEnabled(False)
            self.peak_prominence.setEnabled(False)
            self.volume_button.setEnabled(False)
            self.coverslip_height.setText('')

        self.imshow()

        if reset:
            self.reset_display()
        else:
            # preserve selected cell if tracking info is available
            if hasattr(self, 'selected_particle') and self.selected_particle_n is not None:
                self.select_cell(particle=self.selected_particle_n)
            
            # or clear highlight
            else:
                self.canvas.clear_selection_overlay() # no tracking data, clear highlights

        if len(self.frame.cells)>0 and not hasattr(self.frame.cells[0], 'green'):
            self.get_red_green()
        
        if hasattr(self.frame, 'red_fluor_threshold'):
            self.red_threshold.setText(f'{self.frame.red_fluor_threshold:.2f}')
            self.green_threshold.setText(f'{self.frame.green_fluor_threshold:.2f}')
        else:
            self.red_threshold.setText('')
            self.green_threshold.setText('')
        
        self.update_voxel_size_labels()

        if hasattr(self.frame, 'cell_diameter'):
            self.cell_diameter.setText(f'{self.frame.cell_diameter:.2f}')

        if self.FUCCI_dropdown.currentIndex() != 0:
            self.FUCCI_overlay()

        # frame marker on stat plot
        self.stat_plot_frame_marker.setPos(self.frame_number)
        self.status_frame_number.setText(f'Frame: {frame_number}')

    def update_coordinate_label(self, x=None, y=None):
        ''' Update the status bar with the current cursor coordinates. '''
        if x is None or y is None:
            x, y=self.canvas.cursor_pixels

        coordinates=f"{x}, {y}"
        if hasattr(self, 'zstack_number'):
            coordinates+=f", {self.zstack_number}"
        self.status_coordinates.setText(f"Coordinates: ({coordinates})")
        pixel_value=self.get_pixel_value(x, y)
        if self.is_grayscale:
            pixel_string=f'Gray: {pixel_value[0]}'
        else:
            pixel_string=', '.join(f'{color}: {str(p)}' for color, p in zip(('R','G','B'),pixel_value))
        self.status_pixel_value.setText(pixel_string)
    
    def get_pixel_value(self, x, y):
        ''' Get the pixel value at the current cursor position. '''
        if not self.file_loaded:
            return None, None, None
        elif not hasattr(self, 'frame'): # catch case where file_loaded is mistakenly True due to some exception
            return None, None, None
        img=self.canvas.inverse_transform_image(self.canvas.img_data)

        if x<0 or y<0 or x>=img.shape[1] or y>=img.shape[0]: # outside image bounds
            return None, None, None
        
        if self.is_grayscale:
            return [img[y, x]]

        hidden_channels=np.where(~np.array(self.get_RGB()))[0]
        pixel_value=list(img[y, x])
        for channel in hidden_channels:
            pixel_value[channel]=None
        return pixel_value
    
    def update_cell_label(self, cell_n):
        ''' Update the status bar with the selected cell number. '''
        if cell_n is None:
            self.status_cell.setText("Selected Cell: None")
            self.selected_cell_prompt.setText('')
        else:
            self.status_cell.setText(f"Selected Cell: {cell_n}")
            self.selected_cell_prompt.setText(str(cell_n))

    def update_tracking_ID_label(self, tracking_ID):
        ''' Update the status bar with the current tracking ID. '''
        if tracking_ID is None:
            self.status_tracking_ID.setText("Tracking ID: None")
            self.selected_particle_prompt.setText('')
        else:
            self.status_tracking_ID.setText(f"Tracking ID: {tracking_ID}")
            self.selected_particle_prompt.setText(str(tracking_ID))

    def track_centroids(self):
        ''' Track centroids for the current stack. '''
        if not self.file_loaded:
            return

        tracking_range=self.tracking_range.text()
        memory=self.memory_range.text()
        if memory=='':
            memory=0

        self.statusBar().showMessage(f'Tracking centroids...')
        for frame in self.stack.frames:
            if not frame.has_outlines:
                outlines=utils.outlines_list(frame.masks)
                for cell, outline in zip(frame.cells, outlines):
                    cell.outline=outline
                    cell.get_centroid()
                frame.has_outlines=True

        try:
            if tracking_range=='':
                self.stack.track_centroids(memory=memory)
            else:
                self.stack.track_centroids(search_range=float(tracking_range), memory=memory)
        except Exception as e:
            print(e)
            self.statusBar().showMessage(f'Error tracking centroids: {e}', 4000)
            return
        print(f'Tracked centroids for stack {self.stack.name}')
        self.tracking_range.setText(f'{self.stack.tracking_range:.2f}')
        self.statusBar().showMessage(f'Tracked centroids for stack {self.stack.name}.', 2000)
        self.recolor_tracks()
        self.canvas.draw_masks()

    def recolor_tracks(self):
        # recolor cells so each particle has one color over time
        for frame in self.stack.frames:
            if hasattr(frame, 'stored_mask_overlay'):
                del frame.stored_mask_overlay # remove the mask_overlay attribute to force recoloring
            
        t=self.stack.tracked_centroids
        colors=self.canvas.random_color_ID(t['particle'].max()+1)
        t['color']=colors[t['particle']]
        for frame in self.stack.frames:
            tracked_frame=t[t.frame==frame.frame_number].sort_values('cell_number')
            frame.set_cell_attrs('color_ID', self.canvas.cell_cmap(tracked_frame['color']))

    def LUT_slider_changed(self, event):
        ''' Update the LUTs when the sliders are moved. '''
        if not self.file_loaded:
            self.update_LUT_labels()
            return
        self.normalize_custom_button.setChecked(True)
        self.set_LUTs()

    def particle_from_cell(self, cell_number, frame_number=None):
        if not hasattr(self.stack, 'tracked_centroids'):
            return None
        if frame_number is None:
            frame_number=self.frame_number
        t=self.stack.tracked_centroids
        particle=t[(t.frame==frame_number)&(t.cell_number==cell_number)]['particle']
        if len(particle)==1:
            return particle.item()
        elif len(particle)==0:
            return None
        else:
            raise ValueError(f'Cell {cell_number} has {len(particle)} particles in frame {frame_number}')
    
    def cell_from_particle(self, particle, frame_number=None):
        if not hasattr(self.stack, 'tracked_centroids'):
            return None
        if frame_number is None:
            frame_number=self.frame_number
        t=self.stack.tracked_centroids
        cell=t[(t.frame==frame_number)&(t.particle==particle)]['cell_number']
        if len(cell)==1:
            return cell.item()
        elif len(cell)==0:
            return None
        else:
            raise ValueError(f'Particle {particle} has {len(cell)} cells in frame {frame_number}')
        
    def split_particle_tracks(self):
        new_particle=self.stack.split_particle_track(self.selected_particle_n, self.frame_number)

        if new_particle is None:
            return
        
        self.selected_particle_n=new_particle
        
        # assign a random color to the new particle
        new_color=self.canvas.random_cell_color()
        for cell in self.stack.get_particle(self.selected_particle_n):
            cell.color_ID=new_color
            frame=self.stack.frames[cell.frame]
            if hasattr(frame, 'stored_mask_overlay'):
                self.canvas.add_cell_highlight(cell.n, frame=frame, color=new_color, alpha=self.canvas.masks_alpha, layer='mask')
        
        print(f'Split particle {self.selected_particle_n} at frame {self.frame_number}')
        
        self.also_save_tracking.setChecked(True)
        self.plot_particle_statistic()
        self.highlight_track_ends()

    def merge_particle_tracks(self, first_particle, second_particle):
        if hasattr(self.stack, 'tracked_centroids'):
            if first_particle==second_particle: # same particle, no need to merge
                return
            else:
                first_particle_cell=self.stack.get_particle(first_particle)[0]
                if first_particle_cell.frame==self.frame_number: # merging parent only appears in current frame. Merging not possible.
                    return
                merged_color=first_particle_cell.color_ID
                merged, new_head, new_tail=self.stack.merge_particle_tracks(first_particle, second_particle, self.frame_number)
                self.also_save_tracking.setChecked(True)
                if new_head is not None:
                    new_head_color=self.canvas.random_cell_color()
                    for cell in self.stack.get_particle(new_head):
                        cell.color_ID=new_head_color
                        if hasattr(self.stack.frames[cell.frame], 'stored_mask_overlay'):
                            self.canvas.add_cell_highlight(cell.n, frame=self.stack.frames[cell.frame], color=new_head_color, alpha=self.canvas.masks_alpha, layer='mask')
                if new_tail is not None:
                    new_tail_color=self.canvas.random_cell_color()
                    for cell in self.stack.get_particle(new_tail):
                        cell.color_ID=new_tail_color
                        if hasattr(self.stack.frames[cell.frame], 'stored_mask_overlay'):
                            self.canvas.add_cell_highlight(cell.n, frame=self.stack.frames[cell.frame], color=new_tail_color, alpha=self.canvas.masks_alpha, layer='mask')

                for cell in self.stack.get_particle(merged):
                    cell.color_ID=merged_color
                    if hasattr(self.stack.frames[cell.frame], 'stored_mask_overlay'):
                        self.canvas.add_cell_highlight(cell.n, frame=self.stack.frames[cell.frame], color=merged_color, alpha=self.canvas.masks_alpha, layer='mask')

                print(f'Merged particles {first_particle} and {second_particle} at frame {self.frame_number}')
                self.plot_particle_statistic()
                self.highlight_track_ends()
                current_cell=self.cell_from_particle(merged)
                self.canvas.add_cell_highlight(current_cell, color=merged_color, alpha=self.canvas.masks_alpha, layer='mask')

                new_tail_cell=self.cell_from_particle(new_tail)
                if new_tail_cell is not None:
                    self.canvas.add_cell_highlight(new_tail_cell, color=new_tail_color, alpha=self.canvas.masks_alpha, layer='mask')
            
    def set_LUTs(self):
        ''' Set the LUTs for the image display based on the current slider values. '''
        self.canvas.img.setLevels(self.LUT_slider_values)
        self.update_LUT_labels()

    def update_LUT_labels(self):
        ''' Update the labels next to the LUT sliders with the current values. '''
        for slider, labels in zip(self.LUT_range_sliders, self.LUT_range_labels):
            labels[0].setText(str(slider.value()[0]))
            labels[1].setText(str(slider.value()[1]))
        
    def update_display(self):
        """Redraw the image data with whatever new settings have been applied from the toolbar."""
        if not self.file_loaded:
            return
        self.show_seg_overlay()
        img_data=self.canvas.transform_image(self.frame.img)
        seg_data=self.canvas.transform_image(self.frame.outlines)
        self.canvas.update_display(img_data=img_data, seg_data=seg_data, RGB_checks=self.get_RGB())
        self.normalize()
    
    @property
    def normalize_type(self):
        ''' Get the state of the normalize buttons. Returns the selected button as a string. '''
        button_status=[self.normalize_frame_button.isChecked(), self.normalize_stack_button.isChecked(), self.normalize_custom_button.isChecked()]
        button_names=np.array(['frame', 'stack', 'lut'])
        return button_names[button_status][0]
    
    @normalize_type.setter
    def normalize_type(self, value):
        ''' Set the state of the normalize buttons based on the input string. '''
        button_names=np.array(['frame', 'stack', 'lut'])
        button_status=button_names==value
        self.normalize_frame_button.setChecked(button_status[0])
        self.normalize_stack_button.setChecked(button_status[1])
        self.normalize_custom_button.setChecked(button_status[2])
    
    @property
    def LUT_slider_values(self):
        ''' Get the current values of the LUT sliders. '''
        slider_values=[slider.value() for slider in self.LUT_range_sliders]
        
        return slider_values
    
    @LUT_slider_values.setter
    def LUT_slider_values(self, bounds):
        for slider, bound in zip(self.LUT_range_sliders, bounds):
            if bound[0]==bound[1]: # prevent division by zero
                bound=(0,1)
            slider.blockSignals(True)
            slider.setValue(tuple(bound))
            slider.blockSignals(False)
            self.set_LUTs()

    def set_LUT_slider_ranges(self, ranges):
        for slider, slider_range in zip(self.LUT_range_sliders, ranges):
            slider.blockSignals(True)
            slider.setRange(*slider_range)
            slider.blockSignals(False)
    
    def update_voxel_size(self):
        ''' Update the voxel size when the user changes the text in the voxel size box. '''
        if not self.file_loaded:
            return
        xy_size=self.xy_size.text()
        z_size=self.z_size.text()

        if xy_size=='':
            xy_size=None
        else:
            xy_size=float(xy_size)
        if z_size=='':
            z_size=None
        else:
            z_size=float(z_size)

        self.set_voxel_size(xy_size, z_size)

    def set_voxel_size(self, xy_size=None, z_size=None):
        if xy_size is not None:
            for frame in self.stack.frames:
                frame.scale=xy_size
        if z_size is not None:
            for frame in self.stack.frames:
                frame.z_scale=z_size
        self.update_voxel_size_labels()

        # update right toolbar if necessary
        if 'scaled' in self.histogram_menu.currentText():
            self.plot_histogram()
        if 'scaled' in self.particle_stat_menu.currentText():
            self.plot_particle_statistic()
        if 'scaled' in self.seg_overlay_attr.currentText():
            self.show_seg_overlay()

    
    def update_voxel_size_labels(self):
        ''' Update the labels next to the voxel size boxes with the current values. '''
        if hasattr(self.frame, 'scale'):
            xy_size=self.frame.scale
            self.xy_size.setText(str(xy_size))
        if hasattr(self.frame, 'z_scale'):
            z_size=self.frame.z_scale
            self.z_size.setText(str(z_size))

    def update_normalize_frame(self):
        if not self.file_loaded:
            return
        self.normalize()

    def normalize(self):
        if self.is_grayscale: # single channel
            colors=1
        else:
            colors=3
        
        if self.normalize_type=='frame': # normalize the frame
            if self.is_zstack:
                if hasattr(self.frame, 'bounds'):
                    bounds=self.frame.bounds[self.zstack_number]
                else:
                    zstack_bounds=np.quantile(self.frame.zstack.reshape(self.frame.zstack.shape[0], -1, colors), (0.01, 0.99), axis=1).transpose(1,2,0)
                    self.frame.bounds=zstack_bounds
                    bounds=zstack_bounds[self.zstack_number]
                
            else:
                if hasattr(self.frame, 'bounds') and not self.is_zstack:
                    bounds=self.frame.bounds
                else:
                    bounds=np.quantile(self.canvas.img_data.reshape(-1,colors), (0.01, 0.99), axis=0).T
                    self.frame.bounds=bounds

        elif self.normalize_type=='stack': # normalize the stack
            if hasattr(self.stack, 'bounds'):
                bounds=self.stack.bounds
            else:
                if self.is_zstack:
                    all_imgs=np.array([frame.zstack for frame in self.stack.frames]).reshape(-1, colors)
                else:
                    all_imgs=np.array([frame.img for frame in self.stack.frames]).reshape(-1, colors)
                if len(all_imgs)>1e6:
                    # downsample to speed up calculation
                    all_imgs=all_imgs[::len(all_imgs)//int(1e6)]
                bounds=np.quantile(all_imgs, (0.01, 0.99), axis=0).T
                self.stack.bounds=bounds
        
        else: # custom: use the slider values
            bounds=np.array([slider.value() for slider in self.LUT_range_sliders])
        
        self.LUT_slider_values=bounds

        return bounds
    
    def open_command_line(self):
        # Create a separate window for the command line interface
        self.cli_window = CommandLineWindow(self, self.globals_dict, self.locals_dict)
        self.globals_dict['cli'] = self.cli_window.cli
        self.cli_window.show()

    def select_cell(self, particle=None, cell=None):
        ''' Select a cell or particle by number. '''
        if self.FUCCI_mode: # classifying FUCCI, no cell selection
            return
        
        if cell is not None: # select by cell number
            self.selected_cell_n=cell
            self.selected_particle_n=self.particle_from_cell(cell)
        elif particle is not None: # select by particle number
            self.selected_particle_n=particle
            self.selected_cell_n=self.cell_from_particle(particle)
        else: # clear selection
            self.selected_cell_n=None
            self.selected_particle_n=None
        
        # update labels
        self.update_cell_label(self.selected_cell_n)
        self.update_tracking_ID_label(self.selected_particle_n)
        self.plot_particle_statistic() # put info about the particle in the right toolbar

        self.canvas.clear_selection_overlay()

        if self.selected_cell_n is None:
            self.cell_properties_label.setText('') # clear the cell attributes table
            return
        
        # highlight cell
        self.canvas.add_cell_highlight(self.selected_cell_n, alpha=self.canvas.selected_cell_alpha, color=self.canvas.selected_cell_color)

        # show cell attributes in right toolbar
        if len(self.selected_cell.outline)>0: 
            labels=sorted(self.cell_stat_attrs(self.selected_cell))
            attrs=[getattr(self.selected_cell, attr) for attr in labels]
            cell_attrs_label=create_html_table(labels, attrs)
        else:
            cell_attrs_label=''
        self.cell_properties_label.setText(cell_attrs_label)

    def clear_particle_statistic(self):
        self.particle_stat_plot.clear()
        self.particle_stat_plot.setLabel('left', '')
        self.particle_stat_plot.addItem(self.stat_plot_frame_marker)

    def get_particle_attrs(self):
        if not self.file_loaded:
            return
        current_attr=self.particle_stat_menu.currentText()
        items=[]

        if len(self.frame.cells)>0: # add any valid scalar cell attributes
            common_attrs=self.cell_stat_attrs(self.frame.cells[0])
            for cell in self.frame.cells[1:]:
                common_attrs=common_attrs.intersection(set(dir(cell)))
            items.extend(list(common_attrs))

        items=['Select Cell Attribute']+sorted(items)
        self.particle_stat_menu.blockSignals(True)
        self.particle_stat_menu.clear()
        self.particle_stat_menu.addItems(items)
        self.particle_stat_menu.blockSignals(False)
        current_index=self.particle_stat_menu.findText(current_attr)
        if current_index==-1:
            current_index=0
        self.particle_stat_menu.setCurrentIndex(current_index)

    def plot_particle_statistic(self):
        if not self.file_loaded or not hasattr(self.stack, 'tracked_centroids'):
            return
        measurement=self.particle_stat_menu.currentText()
        
        self.particle_stat_plot.clear() # clear the plot
        self.particle_stat_plot.setLabel('left', measurement)
        if measurement=='Select Cell Attribute':
            return
        
        self.particle_stat_plot.addItem(self.stat_plot_frame_marker) # add the frame marker line
        if self.selected_particle_n is not None:
            color=pg.mkColor(np.array(self.canvas.cell_cmap(0))[:3]*255)
            timepoints=self.stack.get_particle_attr(self.selected_particle_n, 'frame')
            if measurement=='cell_cycle': # fetch up-to-date cell cycle classification
                green, red=np.array(self.stack.get_particle_attr(self.selected_particle_n, ['green', 'red'], fill_value=False)).T
                values=green+2*red
            else:
                values=self.stack.get_particle_attr(self.selected_particle_n, measurement, fill_value=np.nan)
            if np.all(np.isnan(values)): # no data to plot
                return
            self.particle_stat_plot.plot(timepoints, values, pen=color, symbol='o', symbolPen='w', symbolBrush=color, symbolSize=7, width=4)
            self.particle_stat_plot.autoRange()

    def FUCCI_click(self, event, current_cell_n):
        if current_cell_n>=0:
            cell=self.frame.cells[current_cell_n]
            if event.button() == Qt.MouseButton.LeftButton:
                self.classify_cell_cycle(cell, 0)
            if event.button() == Qt.MouseButton.RightButton:
                self.classify_cell_cycle(cell, 1)
            if event.button() == Qt.MouseButton.MiddleButton:
                self.classify_cell_cycle(cell, 2)
        else:
            self.select_cell(None)

    def start_cell_split(self, event):
        self.drawing_cell_split=True
        self.cell_split.clearPoints()

        x, y = self.canvas.get_plot_coords(event.scenePos(), pixels=True)
        # Add the first handle
        self.cell_split.add_vertex(y, x)

    def start_drawing_segmentation(self, event):
        self.drawing_cell_roi=True
        self.cell_roi.clearPoints()

        x, y = self.canvas.get_plot_coords(event.scenePos(), pixels=True)
        # Add the first handle
        self.cell_roi.add_vertex(y, x)
        self.cell_roi.first_handle_pos=np.array((y, x))
        self.cell_roi.last_handle_pos=np.array((y, x))

        self.roi_is_closeable=False

    def on_click(self, event):
        if not self.file_loaded:
            return
        
        x, y = self.canvas.get_plot_coords(event.scenePos(), pixels=True)
        current_cell_n = self.get_cell(x, y)
        
        if self.FUCCI_mode: # cell cycle classification
            self.FUCCI_click(event, current_cell_n)

        else:
            if event.button() == Qt.MouseButton.RightButton:
                if self.drawing_cell_roi:
                    self.close_cell_roi()
                elif self.drawing_cell_split:
                    self.split_cell()

                elif event.modifiers() == Qt.KeyboardModifier.ShiftModifier: # split particles
                    self.selected_particle_n=self.particle_from_cell(current_cell_n)
                    if self.selected_particle_n is not None:
                        self.split_particle_tracks()
                elif event.modifiers() == Qt.KeyboardModifier.AltModifier: # split cells
                    self.start_cell_split(event)
                else: # segmentation
                    self.start_drawing_segmentation(event)

            elif event.button() == Qt.MouseButton.LeftButton:
                # cancel right-click actions
                if self.drawing_cell_roi:
                    self.cell_roi.clearPoints()
                    self.drawing_cell_roi=False
                elif self.drawing_cell_split:
                    self.cell_split.clearPoints()
                    self.drawing_cell_split=False

                # cell selection actions
                if  current_cell_n>=0:
                    if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                        # ctrl click deletes cells
                        self.delete_cell_mask(current_cell_n)
                        self.select_cell(None) # deselect the cell

                    elif event.modifiers() == Qt.KeyboardModifier.AltModifier and self.selected_cell_n is not None:
                        self.merge_cell_masks(self.selected_cell_n, current_cell_n)
                        self.select_cell(cell=self.selected_cell_n) # reselect the merged cell

                    elif event.modifiers() == Qt.KeyboardModifier.ShiftModifier: # merge particles
                        if self.selected_particle_n is not None:
                            second_particle=self.particle_from_cell(current_cell_n)
                            if second_particle is not None: # if a particle is found
                                self.merge_particle_tracks(self.selected_particle_n, second_particle)
                        self.select_cell(cell=current_cell_n)

                    elif current_cell_n==self.selected_cell_n:
                        # clicking the same cell again deselects it
                        self.select_cell(None)

                    else:
                        # select the cell
                        self.select_cell(cell=current_cell_n)

                else: # clicked on background, deselect 
                    self.select_cell(None)

    def classify_cell_cycle(self, cell, classification):
        if classification==0:
            cell.green=not cell.green
        elif classification==1:
            cell.red=not cell.red
        else:
            if cell.green and cell.red:
                cell.green=False
                cell.red=False
            else:
                cell.green=True
                cell.red=True
        
        if self.propagate_FUCCI_checkbox.isChecked():
            if hasattr(self.stack, 'tracked_centroids'):
                particle=self.stack.get_particle(cell)
                for cell_timepoint in particle:
                    if cell_timepoint.frame>cell.frame:
                        cell_timepoint.green=cell.green
                        cell_timepoint.red=cell.red

        if self.FUCCI_mode:
            overlay_color=self.FUCCI_dropdown.currentText().lower()
            if overlay_color=='all':
                color=['none','g','r','orange'][2*cell.red+cell.green]
            elif overlay_color=='green':
                color=['none', 'g'][cell.green]
            elif overlay_color=='red':
                color=['none', 'r'][cell.red]
            else:
                color='none'
            self.canvas.add_cell_highlight(cell.n, alpha=1, color=color, img_type='outlines', layer='FUCCI')

        self.plot_particle_statistic()

    @property
    def selected_cell(self):
        if self.selected_cell_n is None:
            return None
        else:
            return self.frame.cells[self.selected_cell_n]
    
    @property
    def selected_particle(self):
        if self.selected_particle_n is None:
            return None
        else:
            return self.stack.get_particle(self.selected_particle_n)

    def mouse_moved(self, pos):
        ''' Dynamically update the cell mask overlay as the user draws a new cell. '''
        if not self.file_loaded:
            return
        
        if self.drawing_cell_split:
            x, y = self.canvas.get_plot_coords(pos, pixels=True) # position in plot coordinates

            self.cell_split.add_vertex(y, x)

        elif self.drawing_cell_roi:
            x, y = self.canvas.get_plot_coords(pos, pixels=True) # position in plot coordinates
            
            if np.array_equal((y, x), self.cell_roi.last_handle_pos):
                return
            else:
                self.cell_roi.add_vertex(y, x)
                if self.roi_is_closeable:
                    if np.linalg.norm(np.array((y, x))-self.cell_roi.first_handle_pos)<3:
                        self.close_cell_roi()
                        return
                else:
                    if np.linalg.norm(np.array((y, x))-self.cell_roi.first_handle_pos)>3:
                        self.roi_is_closeable=True
        
    def get_cell(self, x, y):
        ''' Get the cell number at a given pixel coordinate. '''
        if x < 0 or y < 0 or x >= self.canvas.img_data.shape[1] or y >= self.canvas.img_data.shape[0]:
            return -1 # out of bounds
        cell_n=self.frame.masks[x, y]
        if cell_n==0:
            return -1
        return cell_n-1
    
    def split_cell(self):
        self.drawing_cell_split=False
        self.split_cell_masks()
        self.cell_split.clearPoints()
        self.select_cell(cell=self.selected_cell_n)
        self.update_display()

    def close_cell_roi(self):
        ''' Close the cell ROI and add the new cell mask to the frame. '''
        self.drawing_cell_roi=False
        enclosed_pixels=self.cell_roi.get_enclosed_pixels()
        # remove pixels outside the image bounds
        enclosed_pixels=enclosed_pixels[(enclosed_pixels[:,0]>=0)&
                                        (enclosed_pixels[:,0]<self.frame.masks.shape[0])&
                                        (enclosed_pixels[:,1]>=0)&
                                        (enclosed_pixels[:,1]<self.frame.masks.shape[1])]
        new_mask_n=self.add_cell_mask(enclosed_pixels)
        if new_mask_n is not False:
            print(f'Added cell {new_mask_n}')
        self.cell_roi.clearPoints()
        self.update_display()

    def random_color(self, n_samples=None):
        ''' Generate random colors for the cell masks. '''
        random_colors=np.random.randint(0, self.canvas.cell_n_colors, size=n_samples)
        colors=np.array(self.canvas.cell_cmap(random_colors))[...,:3]

        return colors
        
    def add_cell_mask(self, enclosed_pixels):
        new_mask_n=self.frame.n_cells # new cell number
        cell_mask=np.zeros_like(self.frame.masks, dtype=bool)
        cell_mask[enclosed_pixels[:,0], enclosed_pixels[:,1]]=True
        new_mask=cell_mask & (self.frame.masks==0)

        if new_mask.sum()<5: # check if the mask is more than 4 pixels (minimum for cellpose to generate an outline)
            return False
        
        self.frame.masks[new_mask]=new_mask_n+1

        cell_color_n=self.canvas.random_color_ID()
        cell_color=self.canvas.cell_cmap(cell_color_n)
        if self.frame.has_outlines:
            outline=utils.outlines_list(new_mask)[0]
        else:
            outline=np.empty((0,2), dtype=int)

        self.frame.outlines[outline[:,1], outline[:,0]]=True
        centroid=np.mean(enclosed_pixels, axis=0)
        self.add_cell(new_mask_n, outline, color_ID=cell_color, centroid=centroid, frame_number=self.frame_number, red=False, green=False)
    
        if hasattr(self.stack, 'tracked_centroids'):
            t=self.stack.tracked_centroids
            new_particle_ID=t['particle'].max()+1
            new_particle=pd.DataFrame([[new_mask_n, centroid[0], centroid[1], self.frame_number, new_particle_ID, cell_color_n]], columns=t.columns, index=[t.index.max()+1]) # TODO: handle situations where tracked_centroids has more or fewer columns
            self.stack.tracked_centroids=pd.concat([t, new_particle])
            self.stack.tracked_centroids=self.stack.tracked_centroids.sort_values(['frame', 'particle'])
            self.also_save_tracking.setChecked(True)

        self.canvas.draw_outlines()
        self.highlight_track_ends()
        self.canvas.add_cell_highlight(new_mask_n, alpha=self.canvas.masks_alpha, color=cell_color, layer='mask')

        self.update_ROIs_label()
        return new_mask_n
    
    def split_cell_masks(self, min_size=5):
        curve_coords=np.array([(p.x(), p.y()) for p in self.cell_split.points]).astype(int)
        next_label = np.max(self.frame.masks) + 1
        
        # Create a binary mask of the curve
        curve_mask = np.zeros_like(self.frame.masks, dtype=bool)
        for i in range(len(curve_coords) - 1):
            rr, cc = draw.line(curve_coords[i][0], curve_coords[i][1],
                            curve_coords[i+1][0], curve_coords[i+1][1])
            # remove out-of-bounds coordinates
            inbound_coords=(rr>=0)&(rr<curve_mask.shape[0])&(cc>=0)&(cc<curve_mask.shape[1])
            rr, cc=rr[inbound_coords], cc[inbound_coords]
            curve_mask[cc, rr] = True
        
        # Find unique labels that intersect with the curve
        intersected_labels = np.unique(self.frame.masks[curve_mask])
        intersected_labels = intersected_labels[intersected_labels != 0]
        
        def find_largest_neighbor_label(component_mask, labels_array, max_iter=50):
            """Find the label of the largest neighboring component."""
            def find_neighbors(component_mask, labels_array):
                """Find the labels of neighboring components."""
                dilated = ndimage.binary_dilation(component_mask)
                neighbor_region = dilated & ~component_mask
                neighbor_labels = labels_array[neighbor_region]
                neighbor_labels = neighbor_labels[neighbor_labels != 0]
                return neighbor_labels, dilated
            
            neighbor_labels=[]
            dilated=component_mask
            counter=0
            while len(neighbor_labels) == 0:
                neighbor_labels, dilated=find_neighbors(dilated, labels_array)
                counter+=1
                if counter>max_iter:
                    raise ValueError("No neighbors found")
                
            # Count occurrences of each neighbor label
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            return unique_labels[np.argmax(counts)]
        
        split=False
        new_masks=self.frame.masks.copy()
        for orig_label in intersected_labels:
            # Get the current region
            region_mask = self.frame.masks == orig_label
            
            # Create temporary binary mask with the curve
            temp_mask = region_mask.copy()
            curve_pixels = curve_mask & region_mask
            temp_mask[curve_pixels] = False
            
            # Label connected components
            labeled_parts, num_features = ndimage.label(temp_mask)
            
            if num_features < 2: # Didn't split the label, move on to next candidate
                continue
            
            # Get sizes of all components
            component_sizes = np.array([np.sum(labeled_parts == i) for i in range(1, num_features + 1)])
            
            num_labels=component_sizes>=min_size

            if np.sum(num_labels)<2: # only one component is above the minimum size
                continue
            
            split=True
            # Find largest component to keep original label
            largest_idx = np.argmax(component_sizes) + 1
            
            # Clear original region
            new_masks[region_mask] = 0
            self.frame.masks[region_mask] = 0 # clear the original mask, to be replaced after split computation
            
            # largest component gets original label
            new_masks[labeled_parts == largest_idx] = orig_label
            
            # Process other components
            other_indices = [i + 1 for i in range(num_features) if i + 1 != largest_idx]
            
            new_labels=[]
            for comp_idx in other_indices: # assign new labels to components above minimum size
                if component_sizes[comp_idx - 1] >= min_size:
                    new_labels.append(next_label)
                    new_masks[labeled_parts == comp_idx] = next_label
                    next_label += 1
                else: # merge small components with their largest neighbors
                    component = labeled_parts == comp_idx
                    try:
                        neighbor_label = find_largest_neighbor_label(component, self.frame.masks)
                    except ValueError:
                        neighbor_label=None

                    if neighbor_label is not None:
                        new_masks[component] = neighbor_label

            print(f'Split cell {orig_label-1}, new labels: {", ".join(str(n-1) for n in new_labels)}')
            
            # Handle curve pixels by assigning them to the most connected component
            curve_points = np.where(curve_pixels)
            for y, x in zip(*curve_points):
                neighborhood_slice = (
                    slice(max(0, y-1), min(self.frame.masks.shape[0], y+2)),
                    slice(max(0, x-1), min(self.frame.masks.shape[1], x+2))
                )
                neighbor_values = self.frame.masks[neighborhood_slice]
                
                # Count neighbors for all possible labels in neighborhood
                unique_neighbors = np.unique(neighbor_values)
                unique_neighbors = unique_neighbors[unique_neighbors != 0]
                
                if len(unique_neighbors) > 0:
                    neighbor_counts = [np.sum(neighbor_values == n) for n in unique_neighbors]
                    new_masks[y, x] = unique_neighbors[np.argmax(neighbor_counts)]
                else:
                    new_masks[y, x] = orig_label

            old_cell=self.frame.cells[orig_label-1]
            old_mask=new_masks==orig_label
            if hasattr(old_cell, '_centroid'):
                del old_cell._centroid
            outline=self.add_outline(old_mask)
            self.frame.masks[old_mask]=orig_label # restore the original label to the largest mask
            if self.frame.has_outlines:
                old_cell.outline=outline
            
            for new_label in new_labels:
                enclosed_pixels=np.array(np.where(new_masks==new_label)).T
                if len(enclosed_pixels)>0:
                    self.add_cell_mask(enclosed_pixels)

        if split:
            del self.frame.stored_mask_overlay

    def add_cell(self, n, outline, color_ID=None, red=False, green=False, frame_number=None, **kwargs):
        if frame_number is None: frame_number=self.frame_number
        self.frame.cells=np.append(self.frame.cells, Cell(n, outline, color_ID=color_ID, red=red, green=green, frame_number=frame_number, parent=self.stack.frames[self.frame_number], **kwargs))
        self.frame.n_cells+=1
    
    def add_outline(self, mask):
        outline=utils.outlines_list(mask)[0]
        self.frame.outlines[outline[:,1], outline[:,0]]=True

        return outline

    def merge_cell_masks(self, cell_n1, cell_n2):
        '''
        merges cell_n2 into cell_n1.
        in practice, this reassigns the cell_n2 mask to cell_n1 and deletes cell_n2.
        '''
        if cell_n1==cell_n2:
            return
        
        if cell_n1>cell_n2:
            selected_cell_n=cell_n1-1
        else:
            selected_cell_n=cell_n1

        # edit frame.masks, frame.outlines
        mask_2=self.frame.masks==cell_n2+1

        self.frame.masks[mask_2]=cell_n1+1 # merge masks

        merged_mask=self.frame.masks==cell_n1+1
        self.frame.outlines[merged_mask]=False # remove both outlines
        outline=self.add_outline(merged_mask) # add merged outline

        # edit merged cell object
        new_cell=self.frame.cells[cell_n1]
        new_cell.outline=outline

        if hasattr(new_cell, '_centroid'):
            del new_cell._centroid
        if hasattr(self.stack, 'tracked_centroids'):
            t=self.stack.tracked_centroids
            t.loc[(t.frame==self.frame_number)&(t.cell_number==cell_n1), ['x','y']]=new_cell.centroid.astype(t['x'].dtype)
            self.also_save_tracking.setChecked(True)

        # add new cell mask to the overlay
        self.canvas.add_cell_highlight(cell_n1, alpha=self.canvas.masks_alpha, color=new_cell.color_ID, layer='mask')

        # purge cell 2
        idx=self.frame.delete_cells([cell_n2])
        self.remove_tracking_data([cell_n2], idx, frame_number=self.frame_number)

        print(f'Merged cell {cell_n2} into cell {cell_n1}')

        self.highlight_track_ends()
        self.canvas.draw_outlines()
        self.select_cell(cell=selected_cell_n)
        self.update_ROIs_label()
        self.update_display()

        self.check_cell_numbers()

    def check_cell_numbers(self):
        ''' for troubleshooting: check if the cell numbers in the frame and the masks align. '''
        cell_number_alignment=np.array([cell.n!=n for n, cell in enumerate(self.frame.cells)])
        if np.any(cell_number_alignment):
            print(f'{np.sum(cell_number_alignment)} cell numbers misalign starting with {np.where(cell_number_alignment)[0][0]}')
        
        mask_number_alignment=np.array([n!=mask_n for n, mask_n in enumerate(fastremap.unique(self.frame.masks))])
        if np.any(mask_number_alignment):
            print(f'{np.sum(mask_number_alignment)} cell masks misalign starting with {np.where(mask_number_alignment)[0][0]}')

    def generate_outlines_list(self):
        if not self.file_loaded:
            return

        if self.segment_on_stack.isChecked():
            frames=self.stack.frames
        else:
            frames=[self.frame]

        for frame in self.progress_bar(frames, desc='Generating outlines'):
            outlines=utils.outlines_list(frame.masks)
            for cell, outline in zip(frame.cells, outlines):
                cell.outline=outline
                cell.get_centroid()
            frame.has_outlines=True
        self.statusBar().showMessage(f'Generated outlines.', 1000)

    def delete_cell_mask(self, cell_n, frame=None, update_display=True):
        if frame is None:
            frame=self.frame

        if frame==self.frame: # remove the cell mask from the mask overlay
            self.canvas.add_cell_highlight(cell_n, alpha=0.5, color='none', img_type='outlines', layer='mask')
        
        idx=frame.delete_cells([cell_n])
        self.remove_tracking_data([cell_n], idx, frame_number=frame.frame_number)

        if update_display:
            print(f'Deleted cell {cell_n} from frame {frame.frame_number}')
            if frame==self.frame:
                if self.selected_cell_n==cell_n:
                    # deselect the removed cell if it was selected
                    self.select_cell(None)
                self.canvas.draw_outlines()
                self.highlight_track_ends()
                self.update_display()
                self.update_ROIs_label()

    def remove_tracking_data(self, cell_numbers, idx, frame_number=None):
        ''' Remove cells from specified frame of the tracking data. Renumber the cell numbers in the frame to align with the new cell masks. '''
        if not hasattr(self.stack, 'tracked_centroids'):
            return
        if frame_number is None:
            frame_number=self.frame_number

        t=self.stack.tracked_centroids
        t.drop(t[(t.frame==frame_number)&np.isin(t.cell_number, cell_numbers)].index, inplace=True)
        
        # remap cell numbers in tracked_centroids
        cell_remap=fastremap.component_map(idx, np.arange(len(idx)))
        t.loc[t.frame==frame_number, 'cell_number']=t.loc[t.frame==frame_number, 'cell_number'].map(cell_remap).astype(t.cell_number.dtype)
        self.also_save_tracking.setChecked(True)
        
    def save_tracking(self, event=None, file_path=None):
        if not self.file_loaded:
            return
        
        if not hasattr(self.stack, 'tracked_centroids'):
            print('No tracking data to save.')
            self.statusBar().showMessage('No tracking data to save.', 1500)
            return
        
        if file_path is None:
            file_path=QFileDialog.getSaveFileName(self, 'Save tracking data as...', filter='*.csv')[0]
            if file_path=='':
                return
        self.stack.tracked_centroids[['cell_number', 'y', 'x', 'frame', 'particle']].to_csv(file_path, index=False)
        print(f'Saved tracking data to {file_path}')
    
    def load_tracking_pressed(self):
        if not self.file_loaded:
            return
        file_path=QFileDialog.getOpenFileName(self, 'Load tracking data...', filter='*.csv')[0]
        if file_path=='':
            return
        
        self.stack.tracked_centroids=self.load_tracking_data(file_path)
        self.stack.tracked_centroids=self.fix_tracked_centroids()
        self.statusBar().showMessage(f'Loaded tracking data from {file_path}', 2000)
        self.propagate_FUCCI_checkbox.setEnabled(True)
        self.recolor_tracks()

    def load_tracking_data(self, file_path):
        tracked_centroids=pd.read_csv(file_path, dtype={'frame':int, 'particle':int, 'cell_number':int}, index_col=False)
        print(f'Loaded tracking data from {file_path}')
        return tracked_centroids

    def save_segmentation(self):
        if not self.file_loaded:
            return

        if self.also_save_tracking.isChecked():
            self.save_tracking(file_path=self.stack.name+'tracking.csv')

        if self.save_stack.isChecked():
            frames_to_save=self.stack.frames

        else:
            frames_to_save=[self.stack.frames[self.frame_number]]

        for frame in self.progress_bar(frames_to_save):
            self.save_frame(frame) # save the frame to the same file path


    def save_as_segmentation(self):
        if not self.file_loaded:
            return

        if self.save_stack.isChecked():
            folder_path=QFileDialog.getExistingDirectory(self, 'Save stack to folder...')
            if folder_path=='':
                return
            for frame in self.progress_bar(self.stack.frames):
                file_path=os.path.join(folder_path, os.path.basename(frame.name))
                self.save_frame(frame, file_path=file_path)
        else:
            file_path=QFileDialog.getSaveFileName(self, 'Save frame as...', filter='*_seg.npy')[0]
            folder_path=Path(file_path).parent
            if file_path=='':
                return
            if not file_path.endswith('_seg.npy'):
                file_path=file_path+'_seg.npy'
            self.save_frame(self.frame, file_path)
        
        if self.also_save_tracking.isChecked():
            self.save_tracking(file_path=folder_path+'/tracking.csv')


    def save_frame(self, frame, file_path=None):
        if file_path is None:
            file_path=frame.name
        if not frame.has_outlines:
            print('Generating outlines...')
            outlines_list=utils.outlines_list(frame.masks)
            for cell, outline in zip(frame.cells, outlines_list):
                cell.outline=outline
                cell.get_centroid()
            frame.has_outlines=True
        
        try: # fetch cell cycle data if available
            self.convert_red_green([frame])
            write_attrs=['cell_cycles']

        except AttributeError:
            write_attrs=[]

        if hasattr(frame, 'zstack'):
            frame.img=frame.zstack[self.zstack_number]

        frame.to_seg_npy(file_path, write_attrs=write_attrs, overwrite_img=True)

        frame.name=file_path
        print(f'Saved frame to {file_path}')
        frame.name=file_path

    def export_csv(self):
        if not self.file_loaded:
            return
        
        for frame in self.stack.frames:
            if not frame.has_outlines:
                print(f'Generating outlines for frame {frame.frame_number}...')
                outlines_list=utils.outlines_list(frame.masks)
                for cell, outline in zip(frame.cells, outlines_list):
                    cell.outline=outline
                    cell.get_centroid()
                frame.has_outlines=True

        # get the data to export
        if hasattr(self.stack, 'tracked_centroids'):
            if not hasattr(self.stack, 'velocities'):
                self.stack.get_velocities()
            df=self.stack.velocities
            if 'color' in df.columns:
                df.drop(columns='color', inplace=True)
        else:
            df=self.stack.centroids()
        
        self.convert_red_green()

        cells=np.concatenate([frame.cells for frame in self.stack.frames])
        attrs=self.cell_scalar_attrs(cells[0])
        attrs=attrs-{'n','frame'} # drop columns already included in the dataframe

        for attr in attrs:
            df[attr]=np.array([getattr(cell, attr) for cell in cells])

        root_path=Path(self.stack.name)/'export.csv'
        dialog=ExportWizard(df, self, root_path.as_posix())
        if dialog.exec():
            # Retrieve data from export dialog
            save_path = dialog.save_path
            checked_attributes = dialog.checked_attributes
        else:
            return
        
        export=df[checked_attributes]
        export.to_csv(save_path, index=False)
        print(f'Saved CSV to {save_path}')

        
    def convert_red_green(self, frames=None):
        ''' convert cell.red, cell.green attributes to FUCCI labeling for the stack.'''
        if frames==None:
            frames=self.stack.frames
        for frame in frames:
            try:
                green, red=np.array(frame.get_cell_attrs(['green', 'red'])).T
            except AttributeError:
                self.get_red_green(frame)
                green, red=np.array(frame.get_cell_attrs(['green', 'red'])).T
            except ValueError: # no cells in the frame
                continue
            frame.cell_cycles=green+2*red
            frame.set_cell_attrs('cycle_stage', frame.cell_cycles)

    def FUCCI_overlay_changed(self):
        if not self.file_loaded:
            return
        
        self.tabbed_menu_widget.blockSignals(True) # manually switch tabs (without triggering tab switch event)
        self.tabbed_menu_widget.setCurrentIndex(1) # switch to the FUCCI tab
        self.current_tab=1
        self.tabbed_menu_widget.blockSignals(False)
        overlay_color=self.FUCCI_dropdown.currentText().lower()
        
        # set RGB mode
        if overlay_color == 'none':
            self.set_RGB(True)
        else:
            if overlay_color == 'all':
                self.set_RGB([True, True, False])
            elif overlay_color == 'red':
                self.set_RGB([True, False, False])
            elif overlay_color == 'green':
                self.set_RGB([False, True, False])
            
            # set overlay mode
            self.outlines_checkbox.setChecked(True)
            self.masks_checkbox.setChecked(False)
        
        self.FUCCI_overlay()

    def FUCCI_overlay(self, event=None):
        """Handle cell cycle overlay options."""
        overlay_color=self.FUCCI_dropdown.currentText().lower()
        if self.tabbed_menu_widget.currentIndex()!=1 or overlay_color=='none':
            self.canvas.clear_FUCCI_overlay() # clear FUCCI overlay during basic selection
            self.FUCCI_mode=False
            return

        else:
            self.select_cell(None)
            self.FUCCI_mode=True
            self.canvas.clear_selection_overlay() # clear basic selection during FUCCI labeling
            if len(self.frame.cells)==0:
                return
            if overlay_color == 'all':
                colors=np.array(['g','r','orange'])
                green, red=np.array(self.frame.get_cell_attrs(['green', 'red'])).T
                colored_cells=np.where(red | green)[0] # cells that are either red or green
                cell_cycle=green+2*red-1
                cell_colors=colors[cell_cycle[colored_cells]] # map cell cycle state to green, red, orange
                self.canvas.highlight_cells(colored_cells, alpha=1, cell_colors=cell_colors, img_type='outlines', layer='FUCCI')

            else:
                colored_cells=np.where(self.frame.get_cell_attrs(overlay_color))[0]
                self.canvas.highlight_cells(colored_cells, alpha=1, color=overlay_color, img_type='outlines', layer='FUCCI')

    def reset_display(self):
        self.drawing_cell_roi=False
        self.drawing_cell_split=False
        self.select_cell(None)
        self.FUCCI_dropdown.setCurrentIndex(0) # clear overlay
        self.seg_overlay_attr.setCurrentIndex(0) # clear attribute overlay
        self.set_RGB(True)
        if not self.is_grayscale:
            self.show_grayscale_checkbox.setChecked(False)
        self.canvas.clear_selection_overlay() # remove any overlays (highlighting, outlines)
        self.canvas.img_plot.autoRange()

    def imshow(self):
        ''' Render any changes to the image data (new file, new frame, new z slice). '''
        self.canvas.draw_outlines()
        self.highlight_track_ends()
        self.update_ROIs_label()
        self.update_display()
        self.show_seg_overlay()
        self.plot_histogram()

    def get_RGB(self):
        if self.is_grayscale:
            return None
        else:
            return [checkbox.isChecked() for checkbox in self.RGB_checkboxes]

    def set_RGB(self, RGB):
        if isinstance(RGB, bool):
            RGB=[RGB]*3
        elif len(RGB)!=3:
            raise ValueError('RGB must be a bool or boolean array of length 3.')
        for checkbox, state in zip(self.RGB_checkboxes, RGB):
            checkbox.setChecked(state)
    
    def keyPressEvent(self, event):
        """Handle key press events (e.g., arrow keys for frame navigation)."""
        if event.key() == Qt.Key.Key_Tab:
            # switch between tabs
            if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                current_tab=self.tabbed_menu_widget.currentIndex()
                self.tabbed_menu_widget.setCurrentIndex((current_tab+1)%self.tabbed_menu_widget.count())
        
        if not self.file_loaded:
            return

        # Ctrl-key shortcuts
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if not self.is_grayscale:
                # FUCCI labeling modes
                if event.key() == Qt.Key.Key_R:
                    if self.FUCCI_dropdown.currentIndex() == 2 and self.FUCCI_mode:
                        self.FUCCI_dropdown.setCurrentIndex(0)
                        self.set_RGB(True)
                    else:
                        self.tabbed_menu_widget.setCurrentIndex(1)
                        self.FUCCI_dropdown.setCurrentIndex(2)
                        self.set_RGB([True, False, False])
                    return
                
                elif event.key() == Qt.Key.Key_G:
                    if self.FUCCI_dropdown.currentIndex() == 1 and self.FUCCI_mode:
                        self.FUCCI_dropdown.setCurrentIndex(0)
                        self.set_RGB(True)
                    else:
                        self.tabbed_menu_widget.setCurrentIndex(1)
                        self.FUCCI_dropdown.setCurrentIndex(1)
                        self.set_RGB([False, True, False])
                    return
                
                elif event.key() == Qt.Key.Key_A:
                    if self.FUCCI_dropdown.currentIndex() == 3 and self.FUCCI_mode:
                        self.FUCCI_dropdown.setCurrentIndex(0)
                        self.set_RGB(True)
                    else:
                        self.tabbed_menu_widget.setCurrentIndex(1)
                        self.FUCCI_dropdown.setCurrentIndex(3)
                        self.set_RGB([True, True, False])
                    return

        # r-g-b toggles
        if not self.is_grayscale:
            if event.key() == Qt.Key.Key_R:
                self.RGB_checkboxes[0].toggle()
            elif event.key() == Qt.Key.Key_G:
                self.RGB_checkboxes[1].toggle()
            elif event.key() == Qt.Key.Key_B:
                self.RGB_checkboxes[2].toggle()

        # segmentation overlay
        if event.key() == Qt.Key.Key_X:
            self.masks_checkbox.toggle()
        elif event.key() == Qt.Key.Key_Z:
            self.outlines_checkbox.toggle()
        elif event.key() == Qt.Key.Key_Delete:
            if self.selected_cell_n is not None:
                self.delete_cell_mask(self.selected_cell_n)
                self.select_cell(None)

        # cancel drawing, loading
        if event.key() == Qt.Key.Key_Escape:
            if self.drawing_cell_roi:
                self.cell_roi.clearPoints()
                self.drawing_cell_roi=False
            if self.drawing_cell_split:
                self.cell_split.clearPoints()
                self.drawing_cell_split=False
            if self.is_iterating: # cancel progress bar iteration
                self.cancel_iter=True
                print('Cancel signal received.')

        # Handle frame navigation with left and right arrow keys
        if event.key() == Qt.Key.Key_Left:
            if self.frame_number > 0:
                self.frame_number -= 1
                self.change_current_frame(self.frame_number)

        elif event.key() == Qt.Key.Key_Right:
            if self.frame_number < len(self.stack.frames) - 1:
                self.frame_number += 1
                self.change_current_frame(self.frame_number)

        # Handle z-stack navigation with up and down arrow keys
        if event.key() == Qt.Key.Key_Up:
            if self.is_zstack and self.zstack_number > 0:
                self.update_zstack_number(self.zstack_number - 1)
        elif event.key() == Qt.Key.Key_Down:
            if self.is_zstack and self.zstack_number < self.stack.zstack_number - 1:
                self.update_zstack_number(self.zstack_number + 1)

    def reset_view(self):
        ''' Reset the view to the original image data. '''
        self.FUCCI_dropdown.setCurrentIndex(0)
        self.set_RGB(True)
        self.canvas.img_plot.autoRange()
        if not self.is_grayscale:
            self.show_grayscale_checkbox.setChecked(False)

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            
            # If it's a widget, delete it
            if item.widget():
                item.widget().deleteLater()
            # If it's a layout, clear and delete it recursively
            elif item.layout():
                self.clear_layout(item.layout())
                item.layout().deleteLater()
            # If it's a spacer, just remove it (no need to delete)
            else:
                del item

    def clear_LUT_sliders(self):
        self.clear_layout(self.slider_layout)
        for slider in self.LUT_range_sliders:
            slider.deleteLater()
        self.LUT_range_sliders.clear()

    def clear_RGB_checkboxes(self):
        self.clear_layout(self.RGB_checkbox_layout)
        for checkbox in self.RGB_checkboxes:
            checkbox.deleteLater()
        self.RGB_checkboxes.clear()

    def add_RGB_checkboxes(self, layout):
        color_channels_layout = QHBoxLayout()
        color_channels_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        color_channels_layout.setSpacing(25)
        self.RGB_checkboxes = [QCheckBox(s, self) for s in ['R', 'G', 'B']]
        for checkbox in self.RGB_checkboxes:
            checkbox.setChecked(True)
            color_channels_layout.addWidget(checkbox)
        self.show_grayscale_checkbox=QCheckBox("Grayscale", self)
        self.show_grayscale_checkbox.setChecked(False)
        layout.addSpacerItem(self.vertical_spacer())
        layout.addLayout(color_channels_layout)
        layout.addWidget(self.show_grayscale_checkbox)
        
        for checkbox in self.RGB_checkboxes:
            checkbox.stateChanged.connect(self.update_display)
        self.show_grayscale_checkbox.stateChanged.connect(self.show_grayscale_toggled)

    def reorder_channels(self):
        if not self.file_loaded:
            return
        if self.is_grayscale:
            return

        # Prompt the user for the channel order
        text, accepted = QInputDialog.getText(self, 'Channel Order', 'Enter channel order (e.g., 1,2,0):')
        
        if accepted and text: # confirmed a non-empty input
            try:
                # Parse the input into a tuple of integers
                channel_order = tuple(map(int, text.split(',')))
                
                if len(channel_order) != 3:
                    raise ValueError("Channel order must have exactly three elements.")
                
                for frame in self.progress_bar(self.stack.frames, desc='Reordering channels'):
                    frame.img = frame.img[..., channel_order]

                    if hasattr(frame, 'zstack'):
                        frame.zstack = frame.zstack[..., channel_order]
                    if hasattr(frame, 'bounds'):
                        frame.bounds = frame.bounds[..., channel_order, :]

                self.update_display()
                self.normalize()
            except ValueError as e:
                # Show an error message if the input is invalid
                QMessageBox.critical(self, 'Invalid Input', str(e))

    def toggle_grayscale(self):
        if not self.file_loaded:
            return
        self.show_grayscale_checkbox.toggle()

    def show_grayscale_toggled(self):
        if not self.file_loaded:
            return
        self.canvas.img.set_grayscale(self.show_grayscale_checkbox.isChecked())

    def open_overlay_settings(self):
        from segmentation_viewer.qt import OverlaySettingsDialog
        overlay_dialog = OverlaySettingsDialog(parent=self.canvas)
        overlay_dialog.settings_applied.connect(self.update_gui_elements)
        if overlay_dialog.exec() == QDialog.DialogCode.Accepted:
            self.update_gui_elements(overlay_dialog.get_settings())

    def update_gui_elements(self, settings):
        if self.canvas.masks_alpha!=settings[2]:
            for frame in self.stack.frames:
                if hasattr(frame, 'stored_mask_overlay'):
                    del frame.stored_mask_overlay
        for attr, setting in zip(['selected_cell_color', 'selected_cell_alpha', 'masks_alpha', 'outlines_color', 'outlines_alpha'], settings):
            setattr(self.canvas, attr, setting)

        self.imshow()
        self.select_cell(cell=self.selected_cell_n)

    def clear_channel_layout(self):
        self.clear_layout(self.segmentation_channels_layout)
    
    def add_channel_layout(self, channel_layout):
        self.membrane_channel=QComboBox(self)
        self.membrane_channel_label=QLabel("Membrane Channel:", self)
        self.membrane_channel.addItems(["Gray", "Red", "Green", "Blue"])
        self.membrane_channel.setCurrentIndex(3)
        self.membrane_channel.setFixedWidth(70)
        self.nuclear_channel=QComboBox(self)
        self.nuclear_channel_label=QLabel("Nuclear Channel:", self)
        self.nuclear_channel.addItems(["None", "Red", "Green", "Blue", "FUCCI"])
        self.nuclear_channel.setFixedWidth(70)

        membrane_tab_layout=QHBoxLayout()
        membrane_tab_layout.setSpacing(10)
        membrane_tab_layout.addWidget(self.membrane_channel_label)
        membrane_tab_layout.addWidget(self.membrane_channel)
        nuclear_layout=QHBoxLayout()
        nuclear_layout.setSpacing(5)
        nuclear_layout.addWidget(self.nuclear_channel_label)
        nuclear_layout.addWidget(self.nuclear_channel)

        channel_layout.addLayout(membrane_tab_layout)
        channel_layout.addLayout(nuclear_layout)

        return channel_layout

    def grayscale_mode(self):
        ''' Hide RGB GUI elements when a grayscale image is loaded. '''
        self.is_grayscale=True
        self.clear_LUT_sliders()
        self.clear_RGB_checkboxes()
        self.add_grayscale_sliders(self.slider_layout)
        self.segmentation_channels_widget.hide()
        self.membrane_channel.setCurrentIndex(0)
        self.nuclear_channel.setCurrentIndex(0)

    def RGB_mode(self):
        ''' Show RGB GUI elements when an RGB image is loaded. '''
        self.is_grayscale=False
        self.clear_LUT_sliders()
        self.clear_RGB_checkboxes()
        self.add_RGB_checkboxes(self.RGB_checkbox_layout)
        self.add_RGB_sliders(self.slider_layout)
        self.segmentation_channels_widget.show()
        self.show_grayscale_checkbox.setChecked(False)
        self.show_grayscale_toggled()

    def labeled_LUT_slider(self, slider_name=None, default_range=(0, 65535)):
        labels_and_slider=QHBoxLayout()
        labels_and_slider.setSpacing(2)
        if slider_name is not None:
            slider_label=QLabel(slider_name)
            labels_and_slider.addWidget(slider_label)
        
        slider=FineScrubQRangeSlider(orientation=Qt.Orientation.Horizontal, parent=self)
        slider.setRange(*default_range)
        slider.setValue(default_range)

        range_labels=[QLineEdit(str(val)) for val in slider.value()]
        for label in range_labels:
            label.setFixedWidth(self.digit_width*6)
            label.setAlignment(Qt.AlignTop)
            label.setValidator(QIntValidator(*default_range))
            label.setStyleSheet("""
                QLineEdit {
                    border: none;
                    background: transparent;
                    padding: 0;
                }
            """)
        range_labels[0].setAlignment(Qt.AlignRight)

        # Connect QLineEdit changes to update the slider value
        def update_min_slider_from_edit():
            min_val=int(range_labels[0].text())
            max_val=int(range_labels[1].text())

            if min_val<slider.minimum():
                slider.setMinimum(min_val)
            elif min_val>max_val:
                min_val=max_val
                range_labels[0].setText(str(min_val))
            slider.setValue((min_val, max_val))
        
        def update_max_slider_from_edit():
            min_val=int(range_labels[0].text())
            max_val=int(range_labels[1].text())

            if max_val>slider.maximum():
                slider.setMaximum(max_val)
            elif max_val<min_val:
                max_val=min_val
                range_labels[1].setText(str(max_val))
            slider.setValue((min_val, max_val))

        range_labels[0].editingFinished.connect(update_min_slider_from_edit)
        range_labels[1].editingFinished.connect(update_max_slider_from_edit)

        # Connect slider value changes to update the QLineEdit text
        def update_edits_from_slider():
            min_val, max_val = slider.value()
            range_labels[0].setText(str(min_val))
            range_labels[1].setText(str(max_val))

        slider.valueChanged.connect(update_edits_from_slider)

        labels_and_slider.addWidget(range_labels[0])
        labels_and_slider.addSpacerItem(QSpacerItem(10, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed))
        labels_and_slider.addWidget(slider)
        labels_and_slider.addSpacerItem(QSpacerItem(10, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed))
        labels_and_slider.addWidget(range_labels[1])

        return labels_and_slider, slider, range_labels

    def add_RGB_sliders(self, layout):
        self.LUT_range_sliders=[]
        self.LUT_range_labels=[]
        
        for label in ['R ', 'G ', 'B ']:
            slider_layout, slider, range_labels=self.labeled_LUT_slider(label)
            layout.addLayout(slider_layout)
            self.LUT_range_sliders.append(slider)
            self.LUT_range_labels.append(range_labels)

            slider.valueChanged.connect(self.LUT_slider_changed)

    def add_grayscale_sliders(self, layout):
        self.LUT_range_sliders=[]
        self.LUT_range_labels=[]
        
        slider_layout, slider, range_labels=self.labeled_LUT_slider()
        layout.addLayout(slider_layout)
        self.LUT_range_sliders.append(slider)
        self.LUT_range_labels.append(range_labels)

        slider.valueChanged.connect(self.LUT_slider_changed)

    # Drag and drop event
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        self.open_stack(natsorted(files))

    def fix_tracked_centroids(self, t=None, inplace=True):
        ''' make sure every cell is accounted for in the tracking data. '''
        if t is None:
            t=self.stack.tracked_centroids
        for frame in self.stack.frames:
            tracked_frame=t[t.frame==frame.frame_number]
            tracked_cells=tracked_frame['cell_number']
            frame_cells=frame.get_cell_attrs('n')

            missing_tracks=set(frame_cells)-set(tracked_cells)
            if len(missing_tracks)>0:
                print(f'tracked_centroids is missing {len(missing_tracks)} cells in frame {frame.frame_number}: {missing_tracks}')
                new_particle_numbers=np.arange(len(missing_tracks))+t['particle'].max()+1
                new_particles=pd.DataFrame([[cell.n, cell.centroid[0], cell.centroid[1], frame.frame_number, particle_number] for cell, particle_number in zip(frame.cells[list(missing_tracks)], new_particle_numbers)], columns=['cell_number', 'y', 'x', 'frame', 'particle'])
                if 'color' in t.columns:
                    new_particles['color']=self.canvas.random_color_ID(len(new_particles))
                t=pd.concat([t, new_particles])

            extra_tracks=set(tracked_cells)-set(frame_cells)
            if len(extra_tracks)>0:
                print(f'tracked_centroids has {len(extra_tracks)} extra tracks in frame {frame.frame_number}: {extra_tracks}')
                t.drop(tracked_frame[tracked_frame.cell_number.isin(extra_tracks)].index, inplace=True)
        
        t=t.sort_values(['frame', 'particle'])
        t['particle']=t.groupby('particle').ngroup() # renumber particles contiguously

        if inplace:
            self.stack.tracked_centroids=t
        return t

    #--------------I/O----------------
    def open_stack(self, files):
        self.stack, tracked_centroids=self.load_files(files)
        if not self.stack:
            return
        self.globals_dict['stack']=self.stack
        
        self.file_loaded=True
        if tracked_centroids is not None:
            self.stack.tracked_centroids=tracked_centroids
            self.stack.tracked_centroids=self.fix_tracked_centroids()
            self.propagate_FUCCI_checkbox.setEnabled(True)
            self.recolor_tracks()
        else:
            self.propagate_FUCCI_checkbox.setChecked(False)
            self.propagate_FUCCI_checkbox.setEnabled(False)

        if len(self.stack.frames)==1:
            out_message=f'Loaded frame {self.stack.frames[0].name}.'
        else:
            out_message=f'Loaded stack {self.stack.name} with {len(self.stack.frames)} frames.'
        print(out_message)
        self.statusBar().showMessage(out_message, 3000)
        
        self.frame=self.stack.frames[0]
        if hasattr(self.frame, 'zstack'):
            self.is_zstack=True
            self.zstack_slider.setVisible(True)
            self.zstack_slider.setRange(0, self.frame.zstack.shape[0]-1)
            self.zstack_slider.setValue(0)
            self.zstack_number=0
        else:
            self.is_zstack=False
            self.zstack_slider.setVisible(False)

        if len(self.stack.frames)>1:
            self.frame_slider.setVisible(True)
        else:
            self.frame_slider.setVisible(False)

        self.frame_slider.setRange(0, len(self.stack.frames)-1)
        self.change_current_frame(0, reset=True) # call frame update explicitly (in case the slider value was already at 0)

        for frame in self.stack.frames:
            frame.has_outlines=True
            
        if self.frame.img.ndim==2: # single channel
            self.grayscale_mode()

        elif self.frame.img.ndim==3: # RGB
            self.RGB_mode()

        else:
            raise ValueError(f'Image has {self.frame.img.ndim} dimensions, must be 2 (grayscale) or 3 (RGB).')

        self.canvas.img_plot.autoRange()

        # reset visual settings
        self.saved_visual_settings=[self.default_visual_settings for _ in range(4)]

        # set slider ranges
        self.normalize()
        
    def open_files(self):
        files = QFileDialog.getOpenFileNames(self, 'Open file(s)', filter='*seg.npy *.tif *.tiff *.nd2')[0]
        if len(files) > 0:
            self.open_stack(files)

    def open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, 'Open folder of segmentation files')
        if folder:
            self.open_stack([folder])

    def load_files(self, files):
        '''
        Load a stack of images. 
        If a tracking.csv is found, the tracking data is returned as well
        '''
        tracked_centroids=None
        tracking_file=None

        #----figure out what's being loaded----
        if os.path.isdir(files[0]): # if a folder is selected, load all files in the folder
            seg_files=[]
            tif_files=[]
            nd2_files=[]

            for f in natsorted(os.listdir(files[0])):
                if f.endswith('seg.npy'):
                    seg_files.append(os.path.join(files[0], f))
                elif f.lower().endswith('tif') or f.lower().endswith('tiff'):
                    tif_files.append(os.path.join(files[0], f))
                elif f.endswith('nd2'):
                    nd2_files.append(os.path.join(files[0], f))
                elif f.endswith('tracking.csv'):
                    tracking_file=os.path.join(files[0], f)

        else: # list of files
            seg_files=[f for f in files if f.endswith('seg.npy')]
            img_files=[f for f in files if f.lower().endswith('tif') or f.lower().endswith('tiff') or f.lower().endswith('nd2')]
            tracking_files=[f for f in files if f.endswith('tracking.csv')]
            if len(tracking_files)>0:
                tracking_file=tracking_files[-1]

        #----load the files----
        # only loads one type of file per call
        # tries to load seg.npy files first, then image files
        if len(seg_files)>0: # segmented file paths
            stack=SegmentedStack(frame_paths=seg_files, load_img=True, progress_bar=self.progress_bar)
            if tracking_file is not None:
                tracked_centroids=self.load_tracking_data(tracking_file)
            self.file_loaded = True
            return stack, tracked_centroids

        elif len(img_files)>0: # image files
            from segmentation_viewer.io import read_image_file
            frames=[]
            for file_path in img_files:
                file_path=Path(file_path)
                imgs=read_image_file(str(file_path), progress_bar=self.progress_bar, desc=f'Loading {file_path.name}')
                if imgs is None:
                    return False, None
                for v, img in enumerate(self.progress_bar(imgs, desc=f'Processing {file_path.stem}')):
                    if img.shape[-1]==2: # pad to 3 color channels
                        img=np.stack([img[..., 0], img[..., 1], np.zeros_like(img[..., 0])], axis=-1)
                    elif img.shape[-1]==1: # single channel
                        img=img[..., 0] # drop the last dimension

                    if len(img)>1: # z-stack
                        frames.append(segmentation_from_zstack(img, name=file_path.with_name(file_path.stem+f'-{v}_seg.npy')))
                    else: # single slice
                        frames.append(segmentation_from_img(img[0], name=file_path.with_name(file_path.stem+f'-{v}_seg.npy')))

            stack=SegmentedStack(from_frames=frames)
            self.file_loaded = True
            return stack, None
        
        else: # can't find any seg.npy or tiff files, ignore
            self.statusBar().showMessage(f'ERROR: File {files[0]} is not a seg.npy or tiff file, cannot be loaded.', 4000)
            return False, None

    def delete_frame(self, event=None, frame_number=None):
        if not self.file_loaded:
            return
        
        if frame_number is None:
            frame_number=self.frame_number

        if len(self.stack.frames)==1:
            return
        
        self.stack.delete_frame(frame_number)
        self.frame_slider.setRange(0, len(self.stack.frames)-1)
        self.change_current_frame(min(frame_number, len(self.stack.frames)-1))

    def make_substack(self):
        if not self.file_loaded:
            return
        # popup dialog to select the range of frames to include in the substack
        dialog=SubstackDialog(len(self.stack.frames), self)
        if dialog.exec() == QDialog.Accepted:
            substack_frames = dialog.get_input()
            if substack_frames is None:
                return
            substack_frames=np.array(substack_frames)
            self.stack.make_substack(substack_frames)
            self.frame_slider.setRange(0, len(self.stack.frames)-1)
            self.change_current_frame(min(self.frame_number, len(self.stack.frames)-1))
    
    #def import_masks(self):
    
    def import_images(self):
        from segmentation_viewer.io import read_image_file
        files = natsorted(QFileDialog.getOpenFileNames(self, 'Open image file(s)', filter='*.tif *.tiff *.nd2')[0])
        # TODO: check if the number of images matches the number of frames in the stack
        if len(files)==0:
            return
        
        imgs=[]
        for file in files:
            name=Path(file).name
            file_imgs=read_image_file(file, progress_bar=self.progress_bar, desc=f'Importing images from {name}')
            if file_imgs is None:
                continue
            imgs.extend(file_imgs)

        if len(imgs)==0: # no images loaded
            return
        
        for img, frame in zip(imgs, self.stack.frames):
            if img.shape[-1]==1: # single channel
                img=img[..., 0] # drop the last dimension
                self.grayscale_mode()
            else: # RGB
                if img.shape[-1]==2: # pad to 3 color channels
                    img=np.stack([img[..., 0], img[..., 1], np.zeros_like(img[..., 0])], axis=-1)
                self.RGB_mode()

            frame.img=img[0]
            if len(img)>1: # z-stack
                self.zstack_number=0 # if any z-stack images are loaded, reset the z-stack number (a little redundant)
                frame.zstack=img
            else: # single slice
                if hasattr(frame, 'zstack'):
                    del frame.zstack

            if hasattr(frame, 'bounds'):
                del frame.bounds

        self.change_current_frame(self.frame_number)

    def get_red_green(self, frame=None):
        ''' Fetch or create red and green attributes for cells in the current frame. '''
        if frame is None:
            frame=self.frame

        for cell in frame.cells:
            if hasattr(cell, 'cycle_stage'):
                cell.green=cell.cycle_stage==1 or cell.cycle_stage==3
                cell.red=cell.cycle_stage==2 or cell.cycle_stage==3
            else:
                cell.red=False
                cell.green=False

    def update_packages(self):
        """
        Update segmentation_tools and segmentation_viewer packages from GitHub.
        Pulls only the src directories and updates the local installations.
        """
        from segmentation_viewer.update import update_packages
        try:
            update_packages()
        except Exception as e:
            QMessageBox.critical(self, 'Update Failed', 
                                    f'Package update error: {str(e)}')
            return
        QMessageBox.information(self, 'Update Complete', 'Packages updated successfully. Please restart the application.')
            
    def closeEvent(self, event):
        # Close the command line window when the main window is closed
        if hasattr(self, 'shape_dialog'):
            self.shape_dialog.close()
        if hasattr(self, 'cli_window'):
            self.cli_window.close()
        event.accept()


def load_stylesheet(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def create_html_table(labels, values):
    if len(labels) != len(values):
        raise ValueError("Labels and values must be of the same length")

    html = """
    <table style="border-collapse: collapse; width: 100%;">
        <thead>
            <tr>
                <th style="text-align: left; padding: 8px; border-bottom: 2px solid #ddd;">Label</th>
                <th style="text-align: left; padding: 8px; border-bottom: 2px solid #ddd;">Value</th>
            </tr>
        </thead>
        <tbody>
    """
    # Loop to add rows
    for label, value in zip(labels, values):
        value=round(value, 2) # round to 2 decimal places
        html += f"""
        <tr>
            <td style="padding: 4px;"><b>{label}:</b></td>
            <td style="padding: 4px;">{value}</td>
        </tr>
        """
    
    html += """
        </tbody>
    </table>
    """
    return html


def main():
    pg.setConfigOptions(useOpenGL=True)
    #pg.setConfigOptions(enableExperimental=True)

    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()

    darktheme_stylesheet=load_stylesheet(importlib.resources.files('segmentation_viewer.assets').joinpath('darktheme.qss'))
    app.setStyleSheet(darktheme_stylesheet)
    app.quitOnLastWindowClosed = True
    ui = MainWidget()
    ui.show()
    app.exec()

if __name__ == '__main__':
    main()