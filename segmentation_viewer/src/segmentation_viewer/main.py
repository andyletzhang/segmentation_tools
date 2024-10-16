import sys
import numpy as np
import pandas as pd
from cellpose import utils # takes a while to import :(
import os

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QComboBox, QPushButton, QRadioButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QCheckBox, QSpacerItem, QSizePolicy, QFileDialog,
    QLineEdit, QTabWidget, QSlider, QGraphicsEllipseItem, QFormLayout, QSplitter, QProgressBar
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIntValidator, QDoubleValidator, QIcon, QFontMetrics
from superqt import QRangeSlider
import pyqtgraph as pg

from segmentation_tools.segmented_comprehension import TimeSeries, Image, HeightMap, Cell
from segmentation_viewer.canvas import PyQtGraphCanvas, CellMaskPolygon
from segmentation_viewer.command_line import CommandLineWindow

import importlib.resources
from tqdm import tqdm

# TODO: fine-scrub LUTs on right click or something
# TODO: add mouse and keyboard shortcuts to interface
# TODO: normalize the summed channels when show_grayscale
# TODO: add num_ROIs to segmentation tab

# TODO: get_mitoses, visualize mitoses, edit mitoses

# TODO: FUCCI tab - show cc occupancies as a stacked bar
# TODO: expand/collapse segmentation plot
# TODO: undo/redo
# TODO: load TIFs
# TODO: segmentation channel 2 "FUCCI" option which blends R and G channels
# TODO: some image pyramid approach to speed up work on large images??

darktheme_stylesheet = """
    QWidget {
        background-color: #2e2e2e;
        color: #ffffff;
    }

    QWidget#bordered {
            border: 2px solid #4b4b4b;
            border-radius: 5px;
            padding: 5px;
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
    
    /* Specific styling for QRangeSlider */
    QRangeSlider {
        background-color: transparent; /* Make the background transparent so it doesn't conflict */
        height: 12px;
    }
"""

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

        # ----------------Toolbar items----------------
        self.spacer = (0,10)

        # Open Buttons
        self.open_button = QPushButton("Open Files", self)
        self.open_folder = QPushButton("Open Folder", self)
        open_menu=QHBoxLayout()
        open_menu.setSpacing(5)
        open_menu.addWidget(self.open_button)
        open_menu.addWidget(self.open_folder)

        # RGB
        self.RGB_checkbox_layout = QVBoxLayout()
        self.add_RGB_checkboxes(self.RGB_checkbox_layout)

        self.slider_layout=QVBoxLayout()
        self.font_metrics=QFontMetrics(QLabel().font()) # metrics for the default font
        self.three_digit_width=self.font_metrics.horizontalAdvance('000')
        self.add_RGB_sliders(self.slider_layout)

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
        self.normalize_frame_button.setChecked(True)
        self.normalize_type='frame'

        # Command Line Interface
        self.command_line_button=QPushButton("Open Command Line", self)
        self.globals_dict = {'main': self, 'np': np}
        self.locals_dict = {}

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

        #----------------Right Toolbar----------------
        self.particle_stat_plot=pg.PlotWidget(title='Tracked Cell Statistics', background='#2e2e2e')
        self.particle_stat_plot.setLabel('bottom', 'Frame')
        self.stat_plot_frame_marker=pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=2))
        self.particle_stat_plot.addItem(self.stat_plot_frame_marker)

        cell_ID_widget=QWidget(objectName='bordered')
        cell_ID_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.cell_ID_layout=QFormLayout(cell_ID_widget)
        selected_cell_label=QLabel("Cell ID:", self)
        self.selected_cell_prompt=QLineEdit(self, placeholderText='None')
        self.selected_cell_prompt.setValidator(QIntValidator(bottom=0)) # non-negative integers only
        selected_particle_label=QLabel("Tracking ID:", self)
        self.selected_particle_prompt=QLineEdit(self, placeholderText='None')
        self.selected_particle_prompt.setValidator(QIntValidator(bottom=0)) # non-negative integers only
        self.cell_ID_layout.addRow(selected_cell_label, self.selected_cell_prompt)
        self.cell_ID_layout.addRow(selected_particle_label, self.selected_particle_prompt)

        particle_stat_widget=QWidget(objectName='bordered')
        particle_stat_layout=QVBoxLayout(particle_stat_widget)
        particle_stat_layout.setSpacing(0)
        particle_stat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.area_button=QRadioButton("Area", self)
        self.perimeter_button=QRadioButton("Perimeter", self)
        self.circularity_button=QRadioButton("Circularity", self)
        self.cell_cycle_button=QRadioButton("Cell Cycle", self)
        self.area_button.setChecked(True)
        particle_stat_layout.addWidget(self.area_button)
        particle_stat_layout.addWidget(self.perimeter_button)
        particle_stat_layout.addWidget(self.circularity_button)
        particle_stat_layout.addWidget(self.cell_cycle_button)

        #----------------Frame Slider----------------
        slider_stylesheet="""
            QSlider::groove:horizontal {
                background: #888;
                border: 1px solid #444;
                height: 16px; /* Groove height */
                margin: 0px;
            }

            QSlider::groove:vertical {
                background: #888;
                border: 1px solid #444;
                width: 16px; /* Groove width */
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
            
            QSlider::handle:vertical {
                background: #ccc;
                border: 1px solid #777;
                height: 40px; /* Handle height */
                width: 16px; /* Handle width */
                margin: -8px 0; /* Adjust positioning to align with groove */
                border-radius: 2px; /* Slightly round edges */
            }
        """
        self.frame_slider=QSlider(Qt.Orientation.Horizontal, self)
        self.zstack_slider=QSlider(Qt.Orientation.Vertical, self)
        self.zstack_slider.setVisible(False)
        # Customize the slider to look like a scroll bar
        self.frame_slider.setFixedHeight(15)  # Make the slider shorter in height
        self.frame_slider.setTickPosition(QSlider.TickPosition.NoTicks)  # No tick marks
        self.frame_slider.setStyleSheet(slider_stylesheet)

        self.zstack_slider.setFixedWidth(15)  # Make the slider shorter in height
        self.zstack_slider.setTickPosition(QSlider.TickPosition.NoTicks)  # No tick marks
        self.zstack_slider.setStyleSheet(slider_stylesheet)

        #----------------Layout----------------
        # Main layout
        main_widget = QSplitter()
        self.setCentralWidget(main_widget)

        self.left_toolbar = QWidget()
        left_toolbar_layout = QVBoxLayout(self.left_toolbar)
        left_toolbar_layout.setSpacing(10)
        left_toolbar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        #self.left_toolbar.setFixedWidth(250)

        self.right_toolbar = QSplitter()
        self.right_toolbar.setOrientation(Qt.Orientation.Vertical)
        right_toolbar_widget = QWidget()
        right_toolbar_layout = QVBoxLayout(right_toolbar_widget)
        right_toolbar_layout.setSpacing(10)
        right_toolbar_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        #self.left_toolbar.setFixedWidth(250)

        self.canvas_widget = QWidget()
        canvas_VBoxLayout = QVBoxLayout(self.canvas_widget)
        canvas_VBoxLayout.setContentsMargins(0, 0, 0, 0)
        canvas_HBoxLayout = QHBoxLayout()
        canvas_HBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.canvas = PyQtGraphCanvas(parent=self)
        self.globals_dict['canvas']=self.canvas

        self.cell_roi = CellMaskPolygon()
        self.cell_roi.last_handle_pos = None
        self.canvas.img_plot.addItem(self.cell_roi)

        canvas_HBoxLayout.addWidget(self.canvas)
        canvas_HBoxLayout.addWidget(self.zstack_slider)
        canvas_VBoxLayout.addLayout(canvas_HBoxLayout)
        canvas_VBoxLayout.addWidget(self.frame_slider)

        main_widget.addWidget(self.left_toolbar)
        main_widget.addWidget(self.canvas_widget)
        main_widget.addWidget(self.right_toolbar)
        main_widget.setSizes([250, 800, 250])

        self.tabbed_menu_widget = QTabWidget()
        self.tabbed_menu_widget.addTab(self.get_segmentation_tab(), "Segmentation")
        self.tabbed_menu_widget.addTab(self.get_FUCCI_tab(), "FUCCI")
        self.tabbed_menu_widget.addTab(self.get_tracking_tab(), "Tracking")

        # Left Toolbar layout
        LUT_widget=QWidget(objectName='bordered')
        LUT_layout=QVBoxLayout(LUT_widget)
        LUT_layout.setSpacing(0)
        LUT_layout.addLayout(self.RGB_checkbox_layout)
        LUT_layout.addItem(self.vertical_spacer())
        LUT_layout.addWidget(self.normalize_label)
        LUT_layout.addWidget(self.normalize_widget)
        LUT_layout.addLayout(self.slider_layout)
        LUT_layout.addWidget(segmentation_overlay_widget)

        left_toolbar_layout.addLayout(open_menu)

        left_toolbar_layout.addWidget(LUT_widget)
        
        left_toolbar_layout.addWidget(self.tabbed_menu_widget)

        left_toolbar_layout.addWidget(self.command_line_button)

        left_toolbar_layout.addWidget(save_widget)

        self.saved_visual_settings=[self.get_visual_settings() for _ in range(self.tabbed_menu_widget.count())]
        self.current_tab=0
        self.FUCCI_mode=False
        # Right Toolbar layout
        self.right_toolbar.addWidget(self.particle_stat_plot)
        right_toolbar_layout.addWidget(cell_ID_widget)
        right_toolbar_layout.addWidget(particle_stat_widget)
        self.right_toolbar.addWidget(right_toolbar_widget)
        self.right_toolbar.setSizes([100, 200])

        #----------------Connections----------------
        self.frame_slider.valueChanged.connect(self.update_frame_number)
        self.zstack_slider.valueChanged.connect(self.update_zstack_number)
        # click event
        self.canvas.img_plot.scene().sigMouseClicked.connect(self.on_click)
        self.canvas.seg_plot.scene().sigMouseClicked.connect(self.on_click)
        # normalize
        self.normalize_frame_button.toggled.connect(self.update_normalize_frame)
        self.normalize_stack_button.toggled.connect(self.update_normalize_frame)
        self.normalize_custom_button.toggled.connect(self.update_normalize_frame)
        # segmentation overlay
        self.masks_checkbox.stateChanged.connect(self.canvas.overlay_masks)
        self.outlines_checkbox.stateChanged.connect(self.canvas.overlay_outlines)
        # command line
        self.command_line_button.clicked.connect(self.open_command_line)
        # save
        self.save_button.clicked.connect(self.save_segmentation)
        self.save_as_button.clicked.connect(self.save_as_segmentation)
        # open
        self.open_button.clicked.connect(self.open_files)
        self.open_folder.clicked.connect(self.open_folder_dialog)
        # switch tabs
        self.tabbed_menu_widget.currentChanged.connect(self.tab_switched)
        # input current cell/particle
        self.selected_cell_prompt.textChanged.connect(self.cell_prompt_changed)
        self.selected_cell_prompt.returnPressed.connect(self.cell_prompt_changed)
        self.selected_particle_prompt.textChanged.connect(self.particle_prompt_changed)
        self.selected_particle_prompt.returnPressed.connect(self.particle_prompt_changed)
        # particle measurements
        self.area_button.toggled.connect(self.plot_particle_statistic)
        self.perimeter_button.toggled.connect(self.plot_particle_statistic)
        self.circularity_button.toggled.connect(self.plot_particle_statistic)
        self.cell_cycle_button.toggled.connect(self.plot_particle_statistic)
    
    def tab_switched(self, index):
        if not self.file_loaded:
            self.current_tab=index
            return
        
        self.saved_visual_settings[self.current_tab]=self.get_visual_settings()
        self.current_tab=index
        self.set_visual_settings(self.saved_visual_settings[index])

        self.highlight_track_ends()
        self.FUCCI_overlay()
    
    def set_visual_settings(self, settings):
        self.set_RGB(settings[0])
        self.normalize_type=settings[1]
        self.masks_checkbox.setChecked(settings[2])
        self.outlines_checkbox.setChecked(settings[3])

    def get_visual_settings(self):
        # retrieve the current visual settings
        return [self.get_RGB(), self.normalize_type, self.masks_checkbox.isChecked(), self.outlines_checkbox.isChecked()]
    
    def get_segmentation_tab(self):
        segmentation_tab=QWidget()
        segmentation_layout=QVBoxLayout(segmentation_tab)
        segmentation_layout.setSpacing(5)
        segmentation_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        segment_frame_widget=QWidget(objectName='bordered')
        segment_frame_layout=QVBoxLayout(segment_frame_widget)
        self.cell_diameter=QLineEdit(self)
        self.cell_diameter.setPlaceholderText('Auto')
        self.cell_diameter.setValidator(QDoubleValidator(bottom=0)) # non-negative floats only
        self.cell_diameter.setFixedWidth(40)
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
        self.gap_size=QLineEdit(self)
        self.gap_size.setPlaceholderText('Auto')
        self.gap_size.setValidator(QIntValidator(bottom=0)) # non-negative integers only
        gap_size_layout.addWidget(gap_size_label)
        gap_size_layout.addWidget(self.gap_size)
        regenerate_remove_layout=QHBoxLayout()
        regenerate_outlines_button=QPushButton("Regenerate Outlines", self)
        clear_masks_button=QPushButton("Clear Masks", self)
        regenerate_remove_layout.addWidget(regenerate_outlines_button)
        regenerate_remove_layout.addWidget(clear_masks_button)
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
        segmentation_utils_layout.addLayout(regenerate_remove_layout)

        segmentation_layout.addWidget(segment_frame_widget)
        segmentation_layout.addWidget(segmentation_utils_widget)

        self.mend_gaps_button.clicked.connect(self.mend_gaps)
        self.remove_edge_masks_button.clicked.connect(self.remove_edge_masks)
        self.cell_diameter.textChanged.connect(self.update_cell_diameter)
        self.cell_diameter_calibrate.clicked.connect(self.calibrate_diameter_pressed)
        self.segment_frame_button.clicked.connect(self.segment_frame_pressed)
        self.segment_stack_button.clicked.connect(self.segment_stack_pressed)
        regenerate_outlines_button.clicked.connect(self.regenerate_outlines_list)
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
        # TODO: can pull this off without replace_segmentation via a series of merges.
        # perhaps slower, but preserves tracking and coloration.
        if not self.file_loaded:
            return
        from segmentation_tools.preprocessing import mend_gaps
        if self.segment_on_stack.isChecked():
            frames=self.stack.frames
        else:
            frames=[self.frame]
        
        gap_size=self.gap_size.text()
        for frame in frames:
            if gap_size=='':
                gap_size=frame.mean_cell_area(scaled=False)/2 # default to half the mean cell area
            frame.masks, _=mend_gaps(frame.masks, gap_size)
            self.replace_segmentation(frame)
        
        self.update_display()

    def remove_edge_masks(self):
        # TODO: can pull this off without replace_segmentation via a series of deletions.
        # perhaps slower, but preserves tracking and coloration.
        if not self.file_loaded:
            return
        if self.segment_on_stack.isChecked():
            frames=self.stack.frames
        else:
            frames=[self.frame]
        
        for frame in frames:
            frame.masks=utils.remove_edge_masks(frame.masks)
            self.replace_segmentation(frame)
        
        self.update_display()
    
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
            from segmentation_tools.preprocessing import combine_FUCCI_channels
            membrane=img[...,channels[0]]
            nuclei=combine_FUCCI_channels(img)
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
                from segmentation_tools.preprocessing import combine_FUCCI_channels
                membrane=frame.img[...,channels[0]]
                nuclei=combine_FUCCI_channels(frame.img)
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

        self.update_display()

    def progress_bar(self, iterable, desc=None):
        if len(iterable) == 1:
            return iterable
        else:
            # Initialize tqdm progress bar
            tqdm_bar = tqdm(iterable, desc=desc)
            
            # Initialize QProgressBar
            qprogress_bar = QProgressBar()
            qprogress_bar.setMaximum(len(iterable))

            # Set size policy to match the status bar width
            qprogress_bar.setFixedHeight(int(self.statusBar().height()*0.8))
            qprogress_bar.setFixedWidth(int(self.statusBar().width()*0.2))

            # Temporarily hide existing permanent status bar widgets
            self.status_coordinates.setVisible(False)
            self.status_pixel_value.setVisible(False)

            self.statusBar().addPermanentWidget(qprogress_bar)

            # Custom iterator to update both progress bars
            def custom_iterator():
                for i, item in enumerate(iterable):
                    yield item
                    tqdm_bar.update(1)
                    qprogress_bar.setValue(i + 1)
                tqdm_bar.close()
                self.statusBar().removeWidget(qprogress_bar)
                # Restore existing permanent status bar widgets
                self.status_coordinates.setVisible(True)
                self.status_pixel_value.setVisible(True)

            return custom_iterator()
        
    def replace_segmentation(self, frame):
        ''' Regenerate the cell outlines and cell objects from the new segmentation masks. '''
        frame.has_outlines=False
        frame.outlines=utils.masks_to_outlines(frame.masks)
        frame.n_cells=np.max(frame.masks)
        frame.cells = np.array([Cell(n, [], frame_number=frame.frame_number) for n in range(frame.n_cells)])
        if hasattr(frame, 'stored_mask_overlay'):
            del frame.stored_mask_overlay

    def get_FUCCI_tab(self):
        FUCCI_tab = QWidget()
        FUCCI_tab_layout = QVBoxLayout(FUCCI_tab)
        FUCCI_tab_layout.setSpacing(10)
        FUCCI_tab_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        measure_FUCCI_widget=QWidget(objectName='bordered')
        measure_FUCCI_layout=QVBoxLayout(measure_FUCCI_widget)

        red_threshold_layout=QHBoxLayout()
        red_threshold_label=QLabel("Red Threshold:", self)
        self.red_threshold=QLineEdit(self)
        self.red_threshold.setFixedWidth(40)
        self.red_threshold.setPlaceholderText('Auto')
        self.red_threshold.setValidator(QDoubleValidator(bottom=0)) # non-negative floats only
        red_threshold_layout.addWidget(red_threshold_label)
        red_threshold_layout.addWidget(self.red_threshold)

        green_threshold_layout=QHBoxLayout()
        green_threshold_label=QLabel("Green Threshold:", self)
        self.green_threshold=QLineEdit(self)
        self.green_threshold.setFixedWidth(40)
        self.green_threshold.setPlaceholderText('Auto')
        self.green_threshold.setValidator(QDoubleValidator(bottom=0)) # non-negative floats only
        green_threshold_layout.addWidget(green_threshold_label)
        green_threshold_layout.addWidget(self.green_threshold)

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
        measure_FUCCI_layout.addSpacerItem(self.vertical_spacer())
        measure_FUCCI_layout.addLayout(FUCCI_button_layout)

        annotate_FUCCI_layout.addLayout(FUCCI_overlay_layout)
        annotate_FUCCI_layout.addWidget(self.FUCCI_checkbox)
        annotate_FUCCI_layout.addSpacerItem(self.vertical_spacer())
        annotate_FUCCI_layout.addWidget(self.propagate_FUCCI_checkbox)
        annotate_FUCCI_layout.addLayout(clear_FUCCI_layout)

        FUCCI_tab_layout.addWidget(measure_FUCCI_widget)
        FUCCI_tab_layout.addWidget(annotate_FUCCI_widget)

        self.FUCCI_dropdown.currentIndexChanged.connect(self.FUCCI_overlay_changed)
        self.FUCCI_checkbox.stateChanged.connect(self.update_display)
        FUCCI_frame_button.clicked.connect(self.measure_FUCCI_frame)
        FUCCI_stack_button.clicked.connect(self.measure_FUCCI_stack)
        self.propagate_FUCCI_checkbox.stateChanged.connect(self.propagate_FUCCI_toggled)
        clear_frame_button.clicked.connect(self.clear_FUCCI_frame_pressed)
        clear_stack_button.clicked.connect(self.clear_FUCCI_stack_pressed)

        return FUCCI_tab
    
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
        if red_threshold=='':
            red_threshold=None
        else:
            red_threshold=float(red_threshold)
        if green_threshold=='':
            green_threshold=None
        else:
            green_threshold=float(green_threshold)
        return red_threshold, green_threshold
    
    def measure_FUCCI(self, frames):
        red_threshold, green_threshold=self.get_FUCCI_thresholds()
        for frame in self.progress_bar(frames, desc='Measuring FUCCI'):
            if not hasattr(frame, 'FUCCI'):
                frame.FUCCI=frame.img[...,0], frame.img[...,1] # use the red and green channels

            frame.measure_FUCCI(red_fluor_threshold=red_threshold, green_fluor_threshold=green_threshold, orange_brightness=1)
            self.get_red_green(frame)
        
        red_threshold=self.frame.red_fluor_threshold
        green_threshold=self.frame.green_fluor_threshold
        self.red_threshold.setText(f'{red_threshold:.2f}')
        self.green_threshold.setText(f'{green_threshold:.2f}')
        self.FUCCI_dropdown.setCurrentIndex(3)
        self.FUCCI_overlay()

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
        # TODO: fix these once add_cell_highlight is generalized to overwrite mask_overlays
        if not self.file_loaded:
            return
        if not hasattr(self.stack, 'tracked_centroids'):
            self.delete_cell_mask(self.selected_cell_n)
            return
        
        particle_n=self.selected_particle_n
        current_frame_n=self.frame_number
        t=self.stack.tracked_centroids

        head_cell_numbers, head_frame_numbers=np.array(t[(t.particle==particle_n)&(t.frame<=current_frame_n)][['cell_number', 'frame']]).T
        for cell_n, frame_n in zip(head_cell_numbers, head_frame_numbers):
            self.delete_cell_mask(cell_n, self.stack.frames[frame_n])

        # reselect the particle
        self.selected_particle_n=particle_n
        self.plot_particle_statistic()

    def delete_particle_tail(self):
        if not self.file_loaded:
            return
        if not hasattr(self.stack, 'tracked_centroids'):
            self.delete_cell_mask(self.selected_cell_n)
            return
        
        particle_n=self.selected_particle_n
        current_frame_n=self.frame_number
        t=self.stack.tracked_centroids

        head_cell_numbers, head_frame_numbers=np.array(t[(t.particle==particle_n)&(t.frame>=current_frame_n)][['cell_number', 'frame']]).T
        for cell_n, frame_n in zip(head_cell_numbers, head_frame_numbers):
            self.delete_cell_mask(cell_n, self.stack.frames[frame_n])

        # reselect the particle
        self.selected_particle_n=particle_n
        self.plot_particle_statistic()

    def delete_particle(self):
        if not self.file_loaded:
            return
        if not hasattr(self.stack, 'tracked_centroids'):
            self.delete_cell_mask(self.selected_cell_n)
            return
        
        particle_n=self.selected_particle_n
        current_frame_n=self.frame_number
        t=self.stack.tracked_centroids

        head_cell_numbers, head_frame_numbers=np.array(t[t.particle==particle_n][['cell_number', 'frame']]).T
        for cell_n, frame_n in zip(head_cell_numbers, head_frame_numbers):
            self.delete_cell_mask(cell_n, self.stack.frames[frame_n])

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

    def wheelEvent(self, event):
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
                self.frame_number = min(self.frame_number + 1, len(self.stack.frames) - 1)
            else: # scroll up = previous frame
                self.frame_number = max(self.frame_number - 1, 0)
            self.frame_slider.setValue(self.frame_number)

    def update_zstack_number(self, zstack_number):
        if not self.file_loaded:
            return
        self.zstack_number = zstack_number
        self.frame.img=self.frame.zstack[self.zstack_number]
        self.imshow(self.frame, reset=False)

    def update_frame_number(self, frame_number):
        if not self.file_loaded:
            return
        self.frame_number = frame_number
        self.frame=self.stack.frames[self.frame_number]
        
        if self.is_zstack:
            self.frame.img=self.frame.zstack[self.zstack_slider.value()]
            self.zstack_slider.setVisible(True)
            self.zstack_slider.setRange(0, self.frame.zstack.shape[0]-1)
            self.is_zstack=True
        else:
            self.zstack_slider.setVisible(False)
            self.is_zstack=False

        self.imshow(self.frame, reset=False)

        # frame marker on stat plot
        self.stat_plot_frame_marker.setPos(self.frame_number)

    def update_coordinate_label(self, x, y):
        ''' Update the status bar with the current cursor coordinates. '''
        self.status_coordinates.setText(f"Coordinates: ({x}, {y})")
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
        img=self.frame.img

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
        t=self.stack.tracked_centroids
        self.selected_particle_n=new_particle
        
        # assign a random color to the new particle
        new_color=self.canvas.random_cell_color()
        for cell in self.stack.get_particle(self.selected_particle_n):
            cell.color_ID=new_color
            if hasattr(self.stack.frames[cell.frame], 'stored_mask_overlay'):
                del self.stack.frames[cell.frame].stored_mask_overlay # TODO: recolor only the new particle by breaking up the add_cell_highlight method

        self.plot_particle_statistic()
        self.highlight_track_ends()
        current_cell=self.cell_from_particle(self.selected_particle_n)
        self.canvas.add_cell_highlight(current_cell, color=new_color, alpha=0.5, layer='mask')

    def merge_particle_tracks(self, first_particle, second_particle):
        if hasattr(self.stack, 'tracked_centroids'):
            if first_particle==second_particle: # same particle, no need to merge
                return
            else:
                merged_color=self.stack.get_particle(first_particle)[0].color_ID
                new_head, new_tail=self.stack.merge_particle_tracks(first_particle, second_particle, self.frame_number)
                if new_head is not None:
                    new_head_color=self.canvas.random_color_ID()
                    for cell in self.stack.get_particle(new_head):
                        cell.color_ID=new_head_color
                if new_tail is not None:
                    new_tail_color=self.canvas.random_color_ID()
                    for cell in self.stack.get_particle(new_tail):
                        cell.color_ID=new_tail_color
                if hasattr(self.stack, 'tracked_centroids'):
                    # renumber the cells in the merged frame
                    t=self.stack.tracked_centroids
                    cell_numbers=np.unique(t[t.frame==self.frame_number]['cell_number'])
                    new_cell_numbers=np.searchsorted(cell_numbers, t.loc[t.frame==self.frame_number, 'cell_number'])
                    t.loc[t.frame==self.frame_number, 'cell_number']=new_cell_numbers.astype(t['cell_number'].dtype)

                for cell in self.stack.get_particle(first_particle):
                    cell.color_ID=merged_color

                print(f'Merged particles {first_particle} and {second_particle}')
                self.plot_particle_statistic()
                self.highlight_track_ends()
                current_cell=self.cell_from_particle(first_particle)
                self.canvas.add_cell_highlight(current_cell, color=merged_color, alpha=0.5, layer='mask')

                new_tail_cell=self.cell_from_particle(new_tail)
                if new_tail_cell is not None:
                    self.canvas.add_cell_highlight(new_tail_cell, color=new_tail_color, alpha=0.5, layer='mask')
            
    def set_LUTs(self):
        ''' Set the LUTs for the image display based on the current slider values. '''
        bounds=self.get_LUT_slider_values()
        self.canvas.img.setLevels(bounds)
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
        self.canvas.update_display(img_data=self.frame.img, seg_data=self.frame.outlines, RGB_checks=self.get_RGB())
        self.normalize()
    
    def get_normalize_buttons(self):
        ''' Get the state of the normalize buttons. Returns the selected button as a string. '''
        button_status=[self.normalize_frame_button.isChecked(), self.normalize_stack_button.isChecked(), self.normalize_custom_button.isChecked()]
        button_names=np.array(['frame', 'stack', 'lut'])
        return button_names[button_status][0]
    
    def get_LUT_slider_values(self):
        ''' Get the current values of the LUT sliders. '''
        slider_values=[slider.value() for slider in self.LUT_range_sliders]
        
        return slider_values
    
    def set_LUT_slider_values(self, bounds):
        for slider, bound in zip(self.LUT_range_sliders, bounds):
            slider.blockSignals(True)
            slider.setValue(tuple(bound))
            slider.blockSignals(False)
            self.set_LUTs()

    def set_LUT_slider_ranges(self, ranges):
        for slider, slider_range in zip(self.LUT_range_sliders, ranges):
            slider.blockSignals(True)
            slider.setRange(*slider_range)
            slider.blockSignals(False)
        
    def update_normalize_frame(self):
        if not self.file_loaded:
            return
        self.normalize_type=self.get_normalize_buttons()
        self.normalize()

    def normalize(self):
        if self.frame.img.ndim==2: # single channel
            colors=1
        else:
            colors=3
        
        if self.normalize_type=='frame': # normalize the frame
            if hasattr(self.frame, 'bounds'):
                bounds=self.frame.bounds
            else:
                bounds=np.quantile(self.frame.img.reshape(-1,colors), (0.01, 0.99), axis=0).T
                self.frame.bounds=bounds

        elif self.normalize_type=='stack': # normalize the stack
            if hasattr(self.stack, 'bounds'):
                bounds=self.stack.bounds
            else:
                all_imgs=np.array([frame.img for frame in self.stack.frames]).reshape(-1, colors)
                if all_imgs.size>1e6:
                    # downsample to speed up calculation
                    random_pixels=np.random.choice(all_imgs.shape[0], size=int(1e6), replace=True)
                    all_imgs=all_imgs[random_pixels]
                bounds=np.quantile(all_imgs, (0.01, 0.99), axis=0).T
                self.stack.bounds=bounds
        
        else: # custom: use the slider values
            bounds=np.array([slider.value() for slider in self.LUT_range_sliders])
        
        self.set_LUT_slider_values(bounds)

        return bounds
    
    def open_command_line(self):
        # Create a separate window for the command line interface
        self.cli_window = CommandLineWindow(self, self.globals_dict, self.locals_dict)
        self.globals_dict['cli'] = self.cli_window.cli
        self.cli_window.show()

    def select_cell(self, particle=None, cell=None):
        ''' Select a cell or particle by number. '''
        if cell is not None:
            self.selected_cell_n=cell
            self.selected_particle_n=self.particle_from_cell(cell)
        elif particle is not None:
            self.selected_particle_n=particle
            self.selected_cell_n=self.cell_from_particle(particle)
        else: # clear selection
            self.selected_cell_n=None
            self.selected_particle_n=None
        
        self.update_cell_label(self.selected_cell_n)
        self.update_tracking_ID_label(self.selected_particle_n)
        
        # put info about the particle in the right toolbar
        self.plot_particle_statistic()

        self.canvas.clear_selection_overlay()
        if not self.FUCCI_mode and self.selected_cell_n is not None: # basic selection, not cell cycle classification
            self.canvas.add_cell_highlight(self.selected_cell_n)

    def clear_particle_statistic(self):
        self.particle_stat_plot.clear()
        self.particle_stat_plot.setLabel('left', '')
        self.particle_stat_plot.addItem(self.stat_plot_frame_marker)

    def plot_particle_statistic(self):
        if not self.file_loaded or not hasattr(self.stack, 'tracked_centroids'):
            return
        # TODO: follow through mitosis?
        measurement=['area', 'perimeter', 'circularity', 'cell_cycle'][self.selected_statistic]
        # clear the plot
        self.particle_stat_plot.clear()
        self.particle_stat_plot.setLabel('left', measurement)
        self.particle_stat_plot.addItem(self.stat_plot_frame_marker) # add the frame marker line

        if self.selected_particle_n is not None:
            color=pg.mkColor(np.array(self.canvas.cell_cmap(self.selected_statistic)[:3])*255)
            timepoints=self.stack.get_particle_attr(self.selected_particle_n, 'frame')
            if measurement=='cell_cycle': # fetch up-to-date cell cycle classification
                green, red=np.array(self.stack.get_particle_attr(self.selected_particle_n, ['green', 'red'], fill_value=False)).T
                values=green+2*red
            else:
                values=self.stack.get_particle_attr(self.selected_particle_n, measurement)
            self.particle_stat_plot.plot(timepoints, values, pen=color, symbol='o', symbolPen='w', symbolBrush=color, symbolSize=7, width=4)

    @property
    def selected_statistic(self):
        stats=[self.area_button.isChecked(), self.perimeter_button.isChecked(), self.circularity_button.isChecked(), self.cell_cycle_button.isChecked()]
        return stats.index(True)

    def FUCCI_click(self, event, current_cell_n):
        if current_cell_n>=0:
            self.select_cell(cell=current_cell_n)
            cell=self.selected_cell
            if event.button() == Qt.MouseButton.LeftButton:
                self.classify_cell_cycle(cell, 0)
            if event.button() == Qt.MouseButton.RightButton:
                self.classify_cell_cycle(cell, 1)
            if event.button() == Qt.MouseButton.MiddleButton:
                self.classify_cell_cycle(cell, 2)
        else:
            self.select_cell(None)

    def segmentation_click(self, event):
        if not self.drawing_cell_roi:
            self.drawing_cell_roi=True
            self.cell_roi.points=[]

            x, y = self.canvas.get_plot_coords(event.scenePos(), pixels=True)
            # Add the first handle
            self.cell_roi.add_vertex(y, x)
            self.cell_roi.first_handle_pos=np.array((y, x))
            self.cell_roi.last_handle_pos=np.array((y, x))

            self.roi_is_closeable=False
            
        else:
            self.close_cell_roi()

    def on_click(self, event):
        # TODO: split this into separate functions for clarity
        if not self.file_loaded:
            return
        
        x, y = self.canvas.get_plot_coords(event.scenePos(), pixels=True)
        current_cell_n = self.get_cell(x, y)
        
        if self.FUCCI_mode: # cell cycle classification
            self.FUCCI_click(event, current_cell_n)

        else:
            if event.button() == Qt.MouseButton.RightButton: 
                if event.modifiers() == Qt.KeyboardModifier.ShiftModifier: # split particles
                    self.selected_particle_n=self.particle_from_cell(current_cell_n)
                    if self.selected_particle_n is not None:
                        self.split_particle_tracks()
                else: # segmentation
                    self.segmentation_click(event)

            elif event.button() == Qt.MouseButton.LeftButton:
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
        return self.frame.cells[self.selected_cell_n]
    
    @property
    def selected_particle(self):
        return self.stack.get_particle(self.selected_particle_n)

    def mouse_moved(self, pos):
        ''' Dynamically update the cell mask overlay as the user draws a new cell. '''
        if self.file_loaded and self.drawing_cell_roi:
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
        cell_n=self.frame.masks[x, y]-1
        return cell_n
    
    def close_cell_roi(self):
        ''' Close the cell ROI and add the new cell mask to the frame. '''
        self.drawing_cell_roi=False
        enclosed_pixels=self.cell_roi.get_enclosed_pixels()
        # remove pixels outside the image bounds
        enclosed_pixels=enclosed_pixels[(enclosed_pixels[:,0]>=0)&
                                        (enclosed_pixels[:,0]<self.frame.masks.shape[0])&
                                        (enclosed_pixels[:,1]>=0)&
                                        (enclosed_pixels[:,1]<self.frame.masks.shape[1])]
        self.add_cell_mask(enclosed_pixels)
        self.cell_roi.clearPoints()
        self.update_display()

    def random_color(self, n_samples=None):
        ''' Generate random colors for the cell masks. '''
        random_colors=np.random.randint(0, self.canvas.cell_n_colors, size=n_samples)
        colors=np.array(self.canvas.cell_cmap(random_colors))[...,:3]

        return colors
        
    def add_cell_mask(self, enclosed_pixels):
        new_mask_n=self.frame.n_cells
        cell_mask=np.zeros_like(self.frame.masks, dtype=bool)
        cell_mask[enclosed_pixels[:,0], enclosed_pixels[:,1]]=True
        new_mask=cell_mask & (self.frame.masks==0)
        self.new_mask=new_mask
        if new_mask.sum()>4: # if the mask is larger than 4 pixels (minimum for cellpose to generate an outline)
            self.frame.masks[new_mask]=new_mask_n+1
            self.frame.n_cells+=1
            print(f'Added cell {new_mask_n}')

            cell_color_n=np.random.randint(0, self.canvas.cell_n_colors)
            cell_color=self.canvas.cell_cmap(cell_color_n)
            outline=utils.outlines_list(new_mask)[0]

            self.frame.outlines[outline[:,1], outline[:,0]]=True
            centroid=np.mean(enclosed_pixels, axis=0)
            self.add_cell(new_mask_n, outline, color_ID=cell_color_n, frame_number=self.frame_number, red=False, green=False)

            if hasattr(self.stack, 'tracked_centroids'):
                t=self.stack.tracked_centroids
                new_particle_ID=t['particle'].max()+1
                new_particle=pd.DataFrame([[new_mask_n, centroid[0], centroid[1], self.frame_number, new_particle_ID, cell_color_n]], columns=t.columns, index=[len(t)])
                self.stack.tracked_centroids=pd.concat([t, new_particle])
                self.stack.tracked_centroids=self.stack.tracked_centroids.sort_values(['frame', 'particle'])

            self.canvas.draw_outlines()
            self.highlight_track_ends()
            self.canvas.add_cell_highlight(new_mask_n, alpha=0.5, color=cell_color, layer='mask')

            self.update_ROIs_label()
            return True
        else:
            return False
        
    def add_cell(self, n, outline, color_ID=None, red=False, green=False, frame_number=None):
        if frame_number is None: frame_number=self.frame_number
        self.frame.cells=np.append(self.frame.cells, Cell(n, outline, color_ID=color_ID, red=red, green=green, frame_number=frame_number))
    
    def add_outline(self, mask):
        outline=utils.outlines_list(mask)[0]
        self.frame.outlines[outline[:,1], outline[:,0]]=True

        return outline

    def merge_cell_masks(self, cell_n1, cell_n2):
        ''' merges cell_n2 into cell_n1. '''
        if cell_n1==cell_n2:
            return

        # edit frame.masks, frame.outlines
        self.frame.masks[self.frame.masks==cell_n2+1]=cell_n1+1 # merge masks
        self.frame.outlines[self.frame.masks==cell_n1+1]=False # remove both outlines
        outline=self.add_outline(self.frame.masks==cell_n1+1) # add merged outline

        # edit merged cell object
        new_cell=self.frame.cells[cell_n1]
        new_cell.outline=outline
        new_cell.centroid=np.mean(np.argwhere(self.frame.masks==cell_n1+1), axis=0)

        # purge cell 2
        self.frame.delete_cell(cell_n2)
        self.remove_tracking_data(cell_n2)

        # add new cell mask to the overlay
        self.canvas.add_cell_highlight(cell_n1, alpha=0.5, color=new_cell.color_ID, layer='mask')

        print(f'Merged cell {cell_n2} into cell {cell_n1}')

        self.highlight_track_ends()
        self.canvas.draw_outlines()
        self.select_cell(cell_n1)
        self.plot_particle_statistic()
        self.update_ROIs_label()
        self.update_display()

    def regenerate_outlines_list(self):
        if not self.file_loaded:
            return

        if self.segment_on_stack.isChecked():
            frames=self.stack.frames
        else:
            frames=[self.frame]

        for frame in self.progress_bar(frames, desc='Regenerating outlines'):
            outlines=utils.outlines_list(frame.masks)
            for cell, outline in zip(frame.cells, outlines):
                cell.outline=outline
        self.statusBar().showMessage('Regenerated outlines.', 1000)

    def delete_cell_mask(self, cell_n, frame=None):
        if frame is None:
            frame=self.frame

        if frame==self.frame:
            self.canvas.add_cell_highlight(cell_n, alpha=0.5, color='none', img_type='outlines', layer='mask')
        
        self.remove_tracking_data(cell_n, frame_number=frame.frame_number)
        frame.delete_cell(cell_n)
        print(f'Deleted cell {cell_n} from frame {frame.frame_number}')

        if frame==self.frame:
            if self.selected_cell_n==cell_n:
                # deselect the removed cell if it was selected
                self.select_cell(None)
                self.plot_particle_statistic() # clear the particle statistic plot
            self.canvas.draw_outlines()
            self.highlight_track_ends()
            self.update_display()
            self.update_ROIs_label()

    def remove_tracking_data(self, cell_number, frame_number=None):
        ''' Remove a cell from one frame of the tracking data. Renumber the cell numbers in the frame to align with the new cell masks. '''
        # TODO: is there a way to only renumber once instead of applying the same algorithm twice in parallel?
        if not hasattr(self.stack, 'tracked_centroids'):
            return
        if frame_number is None:
            frame_number=self.frame_number

        t=self.stack.tracked_centroids
        t.drop(t[(t.frame==frame_number)&(t.cell_number==cell_number)].index, inplace=True)
        t.loc[(t.frame==frame_number)&(t.cell_number==self.stack.frames[frame_number].n_cells), 'cell_number']=cell_number

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
        self.stack.tracked_centroids=self.fix_tracked_centroids(self.stack.tracked_centroids)
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
        
        if self.unsaved: # loaded a new file, need to save the segmentation to a new location
            self.save_as_segmentation()
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
        
        if self.also_save_tracking.isChecked():
            self.save_tracking(file_path=self.stack.name+'tracking.csv')

        if self.save_stack.isChecked():
            folder_path=QFileDialog.getExistingDirectory(self, 'Save stack to folder...')
            if folder_path=='':
                return
            for frame in self.progress_bar(self.stack.frames):
                file_path=os.path.join(folder_path, os.path.basename(frame.name))
                self.save_frame(frame, file_path=file_path)
        else:
            file_path=QFileDialog.getSaveFileName(self, 'Save frame as...', filter='*_seg.npy')[0]
            if file_path=='':
                return
            if not file_path.endswith('_seg.npy'):
                file_path=file_path[:-4]+'_seg.npy'
            self.save_frame(self.frame, file_path)

    def save_frame(self, frame, file_path=None):
        if file_path is None:
            file_path=frame.name
        if not frame.has_outlines:
            print('Generating outlines...')
            outlines_list=utils.outlines_list(frame.masks)
            for cell, outline in zip(frame.cells, outlines_list):
                cell.outline=outline
            frame.has_outlines=True
        
        try: # fetch cell cycle data if available
            self.convert_red_green([frame])
            write_attrs=['cell_cycles']

        except AttributeError:
            write_attrs=[]

        frame.to_seg_npy(file_path, write_attrs=write_attrs)

        frame.name=file_path
        print(f'Saved frame to {file_path}')
        frame.name=file_path
        
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
        self.select_cell(None)
        self.FUCCI_dropdown.setCurrentIndex(0) # clear overlay
        self.set_RGB(True)
        if not self.is_grayscale:
            self.show_grayscale_checkbox.setChecked(False)
        self.canvas.clear_selection_overlay() # remove any overlays (highlighting, outlines)
        self.canvas.img_plot.autoRange()

    def imshow(self, frame, reset=True):
        ''' Change the displayed frame. Run upon loading a new file or advancing the frame slider. '''
        # reset toolbar
        self.status_frame_number.setText(f'Frame: {frame.frame_number}')
        self.frame = frame

        self.globals_dict['frame']=frame
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
        
        if hasattr(self.frame, 'cell_diameter'):
            self.cell_diameter.setText(f'{self.frame.cell_diameter:.2f}')

        if self.FUCCI_dropdown.currentIndex() != 0:
            self.FUCCI_overlay()

        self.canvas.draw_outlines()
        self.highlight_track_ends()
        self.update_ROIs_label()
        self.update_display()

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

            # save (with overwrite)
            if event.key() == Qt.Key.Key_S:
                self.save_segmentation()
                return
            
            # open
            elif event.key() == Qt.Key.Key_O:
                self.open_files()
                return
        
        # ctrl+shift shortcuts
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier:
            if event.key() == Qt.Key.Key_S:
                self.save_as_segmentation()
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

        # reset visuals
        if event.key() == Qt.Key.Key_Escape:
            self.FUCCI_dropdown.setCurrentIndex(0)
            self.set_RGB(True)
            self.canvas.img_plot.autoRange()
            if not self.is_grayscale:
                self.show_grayscale_checkbox.setChecked(False)

        # Handle frame navigation with left and right arrow keys
        if event.key() == Qt.Key.Key_Left:
            if self.frame_number > 0:
                self.frame_number -= 1
                self.frame_slider.setValue(self.frame_number)
                self.imshow(self.stack.frames[self.frame_number], reset=False)

        elif event.key() == Qt.Key.Key_Right:
            if self.frame_number < len(self.stack.frames) - 1:
                self.frame_number += 1
                self.frame_slider.setValue(self.frame_number)
                self.imshow(self.stack.frames[self.frame_number], reset=False)

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
        self.color_channels_layout = QHBoxLayout()
        self.color_channels_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.color_channels_layout.setSpacing(25)
        self.RGB_checkboxes = [QCheckBox(s, self) for s in ['R', 'G', 'B']]
        for checkbox in self.RGB_checkboxes:
            checkbox.setChecked(True)
            self.color_channels_layout.addWidget(checkbox)
        self.show_grayscale_checkbox=QCheckBox("Grayscale", self)
        self.show_grayscale_checkbox.setChecked(False)
        layout.addSpacerItem(self.vertical_spacer())
        layout.addLayout(self.color_channels_layout)
        layout.addWidget(self.show_grayscale_checkbox)
        
        for checkbox in self.RGB_checkboxes:
            checkbox.stateChanged.connect(self.update_display)
        self.show_grayscale_checkbox.stateChanged.connect(self.show_grayscale_toggled)

    def show_grayscale_toggled(self):
        if not self.file_loaded:
            return
        self.canvas.img.set_grayscale(self.show_grayscale_checkbox.isChecked())

    def clear_channel_layout(self):
        self.clear_layout(self.segmentation_channels_layout)
    
    def add_channel_layout(self, channel_layout):
        self.membrane_channel=QComboBox(self)
        self.membrane_channel_label=QLabel("Membrane Channel:", self)
        self.membrane_channel.addItems(["Gray", "Red", "Green", "Blue"])
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
        self.clear_LUT_sliders()
        self.clear_RGB_checkboxes()
        self.add_grayscale_sliders(self.slider_layout)
        self.segmentation_channels_widget.hide()
        self.membrane_channel.setCurrentIndex(0)
        self.nuclear_channel.setCurrentIndex(0)

    def RGB_mode(self):
        ''' Show RGB GUI elements when an RGB image is loaded. '''
        self.clear_LUT_sliders()
        self.clear_RGB_checkboxes()
        self.add_RGB_checkboxes(self.RGB_checkbox_layout)
        self.add_RGB_sliders(self.slider_layout)
        self.segmentation_channels_widget.show()

    def labeled_LUT_slider(self, slider_name=None):
        labels_and_slider=QHBoxLayout()
        if slider_name is not None:
            slider_label=QLabel(slider_name)
            labels_and_slider.addWidget(slider_label)
        
        slider=QRangeSlider(orientation=Qt.Orientation.Horizontal, parent=self)
        slider.setRange(0, 65535)
        slider.setValue((0, 65535))

        range_labels=[QLabel(str(val)) for val in slider.value()]
        for label in range_labels:
            label.setMinimumWidth(self.three_digit_width)
            label.setAlignment(Qt.AlignTop)
        range_labels[0].setAlignment(Qt.AlignRight)
        

        labels_and_slider.addWidget(range_labels[0])
        labels_and_slider.addSpacerItem(QSpacerItem(10, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed))
        labels_and_slider.addWidget(slider)
        labels_and_slider.addSpacerItem(QSpacerItem(10, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed))
        labels_and_slider.addWidget(range_labels[1])

        self.LUT_range_sliders.append(slider)
        self.LUT_range_labels.append(range_labels)

        slider.valueChanged.connect(self.LUT_slider_changed)

        return labels_and_slider

    def add_RGB_sliders(self, layout):
        self.LUT_range_sliders=[]
        self.LUT_range_labels=[]
        
        for label in ['R ', 'G ', 'B ']:
            layout.addLayout(self.labeled_LUT_slider(label))

    def add_grayscale_sliders(self, layout):
        self.LUT_range_sliders=[]
        self.LUT_range_labels=[]
        
        layout.addLayout(self.labeled_LUT_slider())
        #self.set_LUTs()

    # Drag and drop event
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        from natsort import natsorted
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        self.open_stack(natsorted(files))

    def fix_tracked_centroids(self, t):
        ''' make sure every cell is accounted for in the tracking data. '''
        for frame in self.stack.frames:
            tracked_frame=t[t.frame==frame.frame_number]
            tracked_cells=tracked_frame['cell_number']
            tracked_ns=frame.get_cell_attrs('n')

            missing_tracks=set(tracked_ns)-set(tracked_cells)
            if len(missing_tracks)>0:
                print(f'Frame {frame.frame_number} is missing {len(missing_tracks)} cells: {missing_tracks}')
                new_particle_numbers=np.arange(len(missing_tracks))+t['particle'].max()+1
                new_particles=pd.DataFrame([[cell.n, cell.centroid[0], cell.centroid[1], frame.frame_number, particle_number] for cell, particle_number in zip(frame.cells[list(missing_tracks)], new_particle_numbers)], columns=['cell_number', 'y', 'x', 'frame', 'particle'])
                if 'color' in t.columns:
                    new_particles['color']=self.canvas.random_color_ID(len(new_particles))
                t=pd.concat([t, new_particles])

            extra_tracks=set(tracked_cells)-set(tracked_ns)
            if len(extra_tracks)>0:
                print(f'Frame {frame.frame_number} has {len(extra_tracks)} extra tracks: {extra_tracks}')
                t.drop(tracked_frame[tracked_frame.cell_number.isin(extra_tracks)].index, inplace=True)
        
        t=t.sort_values(['frame', 'particle'])
        t['particle']=t.groupby('particle').ngroup() # renumber particles contiguously

        return t

    def open_stack(self, files):
        self.stack, tracked_centroids=self.load_files(files)
        self.globals_dict['stack']=self.stack
        if not self.stack:
            return
        
        self.file_loaded=True
        if tracked_centroids is not None:
            self.stack.tracked_centroids=tracked_centroids
            self.stack.tracked_centroids=self.fix_tracked_centroids(self.stack.tracked_centroids)
            self.propagate_FUCCI_checkbox.setEnabled(True)
            self.recolor_tracks()
        else:
            self.propagate_FUCCI_checkbox.setEnabled(False)

        out_message=f'Loaded stack {self.stack.name} with {len(self.stack.frames)} frames.'
        print(out_message)
        self.statusBar().showMessage(out_message, 3000)
        
        self.frame_number = 0
        self.frame=self.stack.frames[self.frame_number]
        for frame in self.stack.frames:
            frame.has_outlines=True
            
        if self.frame.img.ndim==2: # single channel
            self.is_grayscale=True
            n_colors=1
            self.grayscale_mode()

        elif self.frame.img.ndim==3: # RGB
            self.is_grayscale=False
            n_colors=3
            self.RGB_mode()

        else:
            raise ValueError(f'Image has {self.frame.img.ndim} dimensions, must be 2 (grayscale) or 3 (RGB).')

        self.imshow(self.frame)
        if hasattr(self.frame, 'zstack'):
            self.zstack_slider.setVisible(True)
            self.zstack_slider.setRange(0, self.frame.zstack.shape[0]-1)
            self.zstack_slider.setValue(0)
            self.is_zstack=True
        else:
            self.zstack_slider.setVisible(False)
            self.is_zstack=False

        self.canvas.img_plot.autoRange()
        self.frame_slider.setRange(0, len(self.stack.frames)-1)

        # set slider ranges
        if self.is_zstack:
            all_imgs=np.array([frame.zstack for frame in self.stack.frames]).reshape(-1, n_colors)
        else:
            all_imgs=np.array([frame.img for frame in self.stack.frames]).reshape(-1, n_colors)
        if len(all_imgs)>1e6:
            # downsample to speed up calculation
            random_pixels=np.random.choice(all_imgs.shape[0], size=int(1e6), replace=True)
            all_imgs=all_imgs[random_pixels]
        stack_range=np.array([np.min(all_imgs, axis=0), np.max(all_imgs, axis=0)]).T
        self.stack.min_max=stack_range
        self.set_LUT_slider_ranges(stack_range)
        
    def open_files(self):
        files = QFileDialog.getOpenFileNames(self, 'Open segmentation file', filter='*seg.npy')[0]
        if len(files) > 0:
            self.open_stack(files)

    def open_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, 'Open folder of segmentation files')
        if folder:
            self.open_stack([folder])

    def load_files(self, files):
        '''
        Load a stack of segmented images. 
        If a tracking.csv is found, the tracking data is returned as well
        '''
        # TODO: maybe load things only when they are needed?
        
        self.file_loaded = True
        tracked_centroids=None

        seg_files=[f for f in files if f.endswith('seg.npy')]
        tif_files=[f for f in files if f.endswith('tif') or f.endswith('tiff')]
        if len(seg_files)>0: # segmented file paths
            stack=TimeSeries(frame_paths=seg_files, load_img=True, progress_bar=self.progress_bar)
            for file in files:
                if file.endswith('tracking.csv'):
                    tracked_centroids=self.load_tracking_data(file)
                    print(f'Loaded tracking data from {file}')
            self.unsaved=False
            return stack, tracked_centroids

        elif len(tif_files)>0: # tif files (only checks if no seg.npy files are found)
            from segmentation_tools.io import read_tif, tiffpage_zstack, tiffpage_frame
            from pathlib import Path

            frames=[]
            for file_path in self.progress_bar(tif_files):
                tif_file=read_tif(file_path)
                file_stem=Path(file_path).stem
                for T in self.progress_bar(range(len(tif_file))):
                    if tif_file.shape[1]>1:
                        img=tiffpage_zstack(tif_file, t=T)
                        frames.append(Image(from_zstack=img, img=img[0], file_path=file_stem+f'-{T}_seg.npy'))
                    else:
                        img=tiffpage_frame(tif_file, t=T)
                        frames.append(Image(from_img=img, file_path=file_stem+f'-{T}_seg.npy'))
            stack=TimeSeries(from_frames=frames)
            self.unsaved=True
            return stack, None
        
        else: # no seg.npy files specified, maybe it's a folder of segmented files?
            try:
                stack=TimeSeries(stack_path=files[0], verbose_load=True, progress_bar=self.progress_bar, load_img=True)
                tracking_path=os.path.join(files[0], 'tracking.csv')
                if os.path.exists(tracking_path):
                    tracked_centroids=self.load_tracking_data(tracking_path)
                    print(f'Loaded tracking data from {tracking_path}')
            except FileNotFoundError: # nope, just reject it
                self.statusBar().showMessage(f'ERROR: File {files[0]} is not a seg.npy file, cannot be loaded.', 4000)
                self.unsaved=False
                return False, None

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

    def closeEvent(self, event):
        # Close the command line window when the main window is closed
        if hasattr(self, 'cli_window'):
            self.cli_window.close()
        event.accept()
    
def main():
    #pg.setConfigOptions(useOpenGL=True)
    #pg.setConfigOptions(enableExperimental=True)

    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()

    app.setStyleSheet(darktheme_stylesheet)
    app.quitOnLastWindowClosed = True
    ui = MainWidget()
    ui.show()
    app.exec()

if __name__ == '__main__':
    main()