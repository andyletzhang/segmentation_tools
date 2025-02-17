import importlib.resources
import os
import sys
from pathlib import Path

import time

import fastremap
import numpy as np
import pandas as pd
import pyqtgraph as pg
from cellpose import utils
from natsort import natsorted
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontMetrics, QIcon, QIntValidator
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage
from segmentation_tools.io import segmentation_from_img, segmentation_from_zstack
from segmentation_tools.segmented_comprehension import Cell, SegmentedStack
from segmentation_tools.utils import cell_scalar_attrs
from segmentation_tools.preprocessing import get_quantile
from skimage import draw
from tqdm import tqdm

from .canvas import CellMaskPolygons, CellSplitLines, PyQtGraphCanvas
from .command_line import CommandLineWindow
from .io import ExportWizard
from .qt import CustomComboBox, SubstackDialog
from .scripting import ScriptWindow
from .ui import LeftToolbar, labeled_LUT_slider
from .utils import create_html_table, load_stylesheet
from .workers import BoundsProcessor
from multiprocessing import cpu_count

# high priority
# TODO: generalized data analysis pipeline. Ability to identify any img-shaped attributes in the frame and overlay them a la heights
# ndimage labeled measurements on any of these attributes to create new ones
# TODO: add LUTs to config file
# TODO: export segplot as gif
# TODO: frame histogram should have options for aggregating over frame or stack
# TODO: import masks (and everything else except img/zstack)
# TODO: File -> export heights tif, import heights tif
# TODO: fix segmentation stat LUTs, implement stack LUTs (when possible). Allow floats when appropriate

# low priority
# TODO: unify print statements with status bar messages
# TODO: when cell is clicked, have option to show its entire colormapped track
# TODO: edit mitoses
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

debug_execution_times = False
N_CORES = cpu_count()

class MainWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        # window setup
        self.setWindowTitle('Segmentation Viewer')
        icon_path = importlib.resources.files('segmentation_viewer.assets').joinpath('icon.ico')
        self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1280, 720)
        self.file_loaded = False  # passive mode
        self.drawing_cell_roi = False
        self.drawing_cell_split = False
        self.spacer = (0, 10)  # default spacer size (width, height)
        self.globals_dict = {'main': self, 'np': np, 'pd': pd}
        self.locals_dict = {}
        self.font_metrics = QFontMetrics(QLabel().font())  # metrics for the default font
        self.digit_width = self.font_metrics.horizontalAdvance('0')  # text length scale
        self.cancel_iter = False  # flag to cancel progress bar iteration
        self.is_iterating = False
        self.circle_mask = None
        self.mitosis_mode = 0
        self.bounds_processor = BoundsProcessor(self, n_cores=1)


        # Status bar
        self.status_cell = QLabel('Selected Cell: None', self)
        self.status_frame_number = QLabel('Frame: None', self)
        self.status_tracking_ID = QLabel('Tracking ID: None', self)
        self.status_coordinates = QLabel('Cursor: (x, y)', self)
        self.status_pixel_value = QLabel('R: None, G: None, B: None', self)
        self.statusBar().addWidget(self.status_cell)
        self.statusBar().addWidget(self.status_frame_number)
        self.statusBar().addWidget(self.status_tracking_ID)
        self.statusBar().addPermanentWidget(self.status_coordinates)
        self.statusBar().addPermanentWidget(self.status_pixel_value)

        # ----------------Frame Slider----------------
        self.frame_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.zstack_slider = QSlider(Qt.Orientation.Vertical, self)
        self.frame_slider.setVisible(False)
        self.zstack_slider.setVisible(False)
        # Customize the slider to look like a scroll bar
        self.frame_slider.setFixedHeight(15)  # Make the slider shorter in height
        self.frame_slider.setTickPosition(QSlider.TickPosition.NoTicks)  # No tick marks

        self.zstack_slider.setFixedWidth(15)  # Make the slider shorter in height
        self.zstack_slider.setTickPosition(QSlider.TickPosition.NoTicks)  # No tick marks

        # ----------------Layout----------------
        # Main layout
        main_widget = QSplitter()
        self.setCentralWidget(main_widget)
        self._init_actions()

        self.canvas_widget = QWidget()
        canvas_VBoxLayout = QVBoxLayout(self.canvas_widget)
        canvas_VBoxLayout.setSpacing(0)
        canvas_VBoxLayout.setContentsMargins(0, 0, 0, 0)
        canvas_HBoxLayout = QHBoxLayout()
        canvas_HBoxLayout.setSpacing(0)
        canvas_HBoxLayout.setContentsMargins(0, 0, 0, 0)

        self.canvas = PyQtGraphCanvas(parent=self)
        self.globals_dict['canvas'] = self.canvas
        self.right_toolbar = self._get_right_toolbar()
        self.left_toolbar = LeftToolbar(self)

        self.cell_roi = CellMaskPolygons(parent=self.canvas)
        self.cell_roi.last_handle_pos = None
        self.canvas.img_plot.addItem(self.cell_roi.img_poly)
        self.canvas.seg_plot.addItem(self.cell_roi.seg_poly)

        self.cell_split = CellSplitLines(parent=self.canvas)
        self.canvas.img_plot.addItem(self.cell_split.img_line)
        self.canvas.seg_plot.addItem(self.cell_split.seg_line)

        canvas_HBoxLayout.addWidget(self.canvas)
        canvas_HBoxLayout.addWidget(self.zstack_slider)
        canvas_VBoxLayout.addLayout(canvas_HBoxLayout)
        canvas_VBoxLayout.addWidget(self.frame_slider)

        main_widget.addWidget(self.left_toolbar)
        main_widget.addWidget(self.canvas_widget)
        main_widget.addWidget(self.right_toolbar)
        main_widget.setCollapsible(1, False)
        main_widget.setSizes([250, 800, 250])

        self.default_visual_settings = self.visual_settings
        self.default_visual_settings['LUTs'] = None
        self.left_toolbar.saved_visual_settings = [self.default_visual_settings for _ in range(4)]
        self.FUCCI_mode = False

        self._load_config()
        self._apply_overlay_settings()

        # ----------------Connections----------------
        self.frame_slider.valueChanged.connect(self.change_current_frame)
        self.zstack_slider.valueChanged.connect(self.update_zstack_number)

        # click event
        self.canvas.img_plot.scene().sigMouseClicked.connect(self._on_click)
        self.canvas.seg_plot.scene().sigMouseClicked.connect(self._on_click)

    def _init_actions(self):
        from .utils import create_action

        file_actions = {
            'Open File(s)': (self._open_files, 'Ctrl+O'),
            'Open Folder': (self._open_folder, 'Ctrl+Shift+O'),
            'Save': (self._save, 'Ctrl+S'),
            'Save As...': (self._save_as, 'Ctrl+Shift+S'),
            'Export CSV...': (self._export_csv, 'Ctrl+Shift+E'),
            'Import Images...': (self.import_images, None),
            'Window Screenshot': (self.save_screenshot, None),
            'Save Stack GIF': (self._save_stack_gif, None),
            'Exit': (self.close, 'Ctrl+Q'),
        }
        edit_actions = {
            'Clear Masks': (self.clear_masks, None),
            'Generate Outlines': (self._generate_outlines, None),
            'Mend Gaps': (self._mend_gaps, None),
            'Remove Edge Masks': (self._remove_edge_masks, None),
        }
        view_actions = {
            'Reset View': (self._reset_view, None),
            'Show Grayscale': (self._toggle_grayscale, None),
            'Invert Contrast': (self._toggle_inverted, 'I'),
            'Overlay Settings...': (self._open_overlay_settings, None),
            'Change LUTs...': (self._change_LUTs, None),
        }
        image_actions = {
            'Reorder Channels...': (self._reorder_channels, None),
            'Rotate Stack Clockwise': (self.rotate_clockwise, None),
            'Rotate Stack Counterclockwise': (self.rotate_counterclockwise, None),
        }
        stack_actions = {'Delete Frame': (self.delete_frame, None), 'Make Substack...': (self.make_substack, None)}
        scripts_actions = {
            'Open Command Line': (self._open_command_line, None),
            'Open Script Editor': (self._open_script_editor, None),
        }
        help_actions = {'Pull Updates': (self._update_packages, None)}
        other_actions = {
            # Keyboard Shortcuts
            'Add Mitosis': (self._add_mitosis, 'Ctrl+M'),
            'Delete Mitosis': (self._delete_mitosis, 'Ctrl+Shift+M'),
            # Other Actions
            'Calibrate Diameter': (self._calibrate_diameter, None),
            'Segment Frame': (self._segment_frame, None),
            'Segment Stack': (self._segment_stack, None),
            'Clear FUCCI Frame': (self._clear_FUCCI_frame, None),
            'Clear FUCCI Stack': (self._clear_FUCCI_stack, None),
            'Measure Volumes': (self._measure_volumes, None),
            'Measure Heights': (self._measure_heights, None),
            'Get Coverslip Height': (self._calibrate_coverslip_height, None),
            'Get Spherical Volumes': (self._compute_spherical_volumes, None),
            'Get Mitoses': (self._get_mitoses, None),
            'Apply Mitosis Weights': (self._apply_new_weights, None),
            'Track Centroids': (self._track_centroids, None),
            'Load Tracking': (self._load_tracking, None),
        }

        # Create Menu Bar
        for menu, actions in zip(
            ['File', 'Edit', 'View', 'Image', 'Stack', 'Scripts', 'Help'],
            [file_actions, edit_actions, view_actions, image_actions, stack_actions, scripts_actions, help_actions],
        ):
            current_menu = self.menuBar().addMenu(menu)
            for key, (function, shortcut) in actions.items():
                action = create_action(key, function, self, shortcut)
                setattr(self, f'{function.__name__}_action', action)
                current_menu.addAction(action)

        # Create Other Actions
        for key, (function, shortcut) in other_actions.items():
            action = create_action(key, function, self, shortcut)
            setattr(self, f'{function.__name__}_action', action)
            self.addAction(action)

    def _load_config(self):
        from platformdirs import user_config_dir

        config_path = Path(user_config_dir('segmentation_viewer')).joinpath('config.yaml')
        if config_path.exists():
            import yaml

            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self._set_config(config)
                return

            except Exception as e:
                print(f'Error loading config file: {e} - using defaults instead.')

        print(f'Creating config file at {config_path}')
        config = self._dump_config(config_path)
        self._set_config(config)

    def _set_config(self, config):
        self.canvas.dark_overlay_settings = config['overlay_settings']
        self.canvas.light_overlay_settings = config['inverted_overlay_settings']
        self.left_toolbar.inverted_checkbox.setChecked(config['inverted'])

    def _dump_config(self, config_path=None):
        import yaml
        from platformdirs import user_config_dir

        config_path = Path(user_config_dir('segmentation_viewer')).joinpath('config.yaml')
        inverted = self.left_toolbar.inverted_checkbox.isChecked()

        current_overlay_settings = {
            attr: getattr(self.canvas, attr)
            for attr in ['selected_cell_color', 'selected_cell_alpha', 'outlines_color', 'outlines_alpha', 'masks_alpha']
        }

        overlay_settings = getattr(self.canvas, 'dark_overlay_settings', current_overlay_settings.copy())
        inverted_overlay_settings = getattr(self.canvas, 'light_overlay_settings', current_overlay_settings.copy())

        config = {
            'overlay_settings': overlay_settings,
            'inverted_overlay_settings': inverted_overlay_settings,
            'inverted': inverted,
        }
        # create the config directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        return config

    def _open_overlay_settings(self):
        from segmentation_viewer.qt import OverlaySettingsDialog

        self.overlay_dialog = OverlaySettingsDialog(parent=self.canvas)
        self.overlay_dialog.settings_applied.connect(self._apply_overlay_settings)
        if self.overlay_dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_overlay_settings(self.overlay_dialog.get_settings())

    def _apply_overlay_settings(self, settings=None):
        inverted = self.left_toolbar.inverted_checkbox.isChecked()
        if settings is None:
            if inverted:
                settings = self.canvas.light_overlay_settings
            else:
                settings = self.canvas.dark_overlay_settings
        else:
            if inverted:
                self.canvas.light_overlay_settings = settings
            else:
                self.canvas.dark_overlay_settings = settings

        redraw_masks = self.canvas.masks_alpha != settings['masks_alpha']

        for attr, setting in settings.items():
            setattr(self.canvas, attr, setting)

        if self.file_loaded:
            if redraw_masks:
                for frame in self.stack.frames:
                    if hasattr(frame, 'stored_mask_overlay'):
                        del frame.stored_mask_overlay
                self.canvas.draw_masks_parallel()

            self.imshow()
            self.select_cell(cell=self.selected_cell_n)

    def _get_right_toolbar(self):
        self.stat_tabs = QTabWidget()
        self.stat_tabs.addTab(self._get_histogram_tab(), 'Histogram')
        self.stat_tabs.addTab(self._get_particle_stat_tab(), 'Particle')
        self.stat_tabs.addTab(self._get_time_series_tab(), 'Time Series')
        self.last_stat_tab = 0

        stat_overlay_widget = QWidget(objectName='bordered')
        stat_overlay_layout = QVBoxLayout(stat_overlay_widget)
        stat_overlay_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        seg_overlay_layout = QHBoxLayout()
        self.seg_overlay_label = QLabel('Overlay Statistic:', self)
        self.seg_overlay_attr = CustomComboBox(self)
        self.seg_overlay_attr.addItems(['Select Cell Attribute'])
        seg_overlay_layout.addWidget(self.seg_overlay_label)
        seg_overlay_layout.addWidget(self.seg_overlay_attr)

        normalize_label = QLabel('Overlay LUTs:', self)
        normalize_widget = QWidget()
        normalize_layout = QHBoxLayout(normalize_widget)
        normalize_layout.setContentsMargins(0, 0, 0, 0)
        self.stat_frame_button = QRadioButton('Frame', self)
        self.stat_stack_button = QRadioButton('Stack', self)
        self.stat_custom_button = QRadioButton('LUT', self)
        normalize_layout.addWidget(self.stat_frame_button)
        normalize_layout.addWidget(self.stat_stack_button)
        normalize_layout.addWidget(self.stat_custom_button)
        self.stat_frame_button.setChecked(True)
        self.stat_LUT_type = 'frame'
        slider_layout, self.stat_LUT_slider, self.stat_range_labels = labeled_LUT_slider(
            default_range=(0, 255), parent=stat_overlay_widget
        )

        cell_ID_widget = QWidget(objectName='bordered')
        self.cell_ID_layout = QFormLayout(cell_ID_widget)
        selected_cell_label = QLabel('Cell ID:', self)
        self.selected_cell_prompt = QLineEdit(self, placeholderText='None')
        self.selected_cell_prompt.setValidator(QIntValidator(bottom=0))  # non-negative integers only
        selected_particle_label = QLabel('Tracking ID:', self)
        self.selected_particle_prompt = QLineEdit(self, placeholderText='None')
        self.selected_particle_prompt.setValidator(QIntValidator(bottom=0))  # non-negative integers only
        self.cell_properties_label = QLabel(self)
        self.cell_ID_layout.addRow(selected_cell_label, self.selected_cell_prompt)
        self.cell_ID_layout.addRow(selected_particle_label, self.selected_particle_prompt)
        self.cell_ID_layout.addRow(self.cell_properties_label)

        stat_overlay_layout.addLayout(seg_overlay_layout)
        stat_overlay_layout.addWidget(normalize_label)
        stat_overlay_layout.addWidget(normalize_widget)
        stat_overlay_layout.addLayout(slider_layout)

        # Create a container widget for all content
        particle_stat_layout = QSplitter(Qt.Orientation.Vertical)
        particle_stat_layout.setContentsMargins(5, 10, 10, 10)
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

        # ----connections-----
        self.stat_tabs.currentChanged.connect(self._stat_tab_switched)
        # cell selection
        self.selected_cell_prompt.textChanged.connect(self._cell_prompt_changed)
        self.selected_cell_prompt.returnPressed.connect(self._cell_prompt_changed)
        self.selected_particle_prompt.textChanged.connect(self._particle_prompt_changed)
        self.selected_particle_prompt.returnPressed.connect(self._particle_prompt_changed)
        # stat overlay
        self.seg_overlay_attr.dropdownOpened.connect(self._get_overlay_attrs)
        self.seg_overlay_attr.activated.connect(self._new_seg_overlay)
        self.seg_overlay_attr.currentIndexChanged.connect(self._new_seg_overlay)
        self.stat_LUT_slider.valueChanged.connect(self._stat_LUT_slider_changed)
        self.stat_frame_button.toggled.connect(self._update_stat_LUT)
        self.stat_stack_button.toggled.connect(self._update_stat_LUT)
        self.stat_custom_button.toggled.connect(self._update_stat_LUT)
        return right_scroll_area

    def _stat_tab_switched(self, index):
        if not self.file_loaded:
            return

        last_tab_combo = [self.histogram_menu, self.particle_stat_menu, self.time_series_menu][self.last_stat_tab]
        current_tab_combo = [self.histogram_menu, self.particle_stat_menu, self.time_series_menu][index]
        current_attr = last_tab_combo.currentText()

        current_tab_combo.changeToText(current_attr)

        self.last_stat_tab = index

    def _get_histogram_tab(self):
        frame_histogram_widget = QWidget()
        frame_histogram_layout = QVBoxLayout(frame_histogram_widget)
        histogram_menu_layout = QHBoxLayout()
        self.histogram_menu = CustomComboBox(self)
        self.histogram_menu.addItems(['Select Cell Attribute'])
        histogram_menu_layout.addWidget(self.histogram_menu)
        histogram_menu_layout.setContentsMargins(40, 0, 0, 0)  # indent the title/menu
        self.histogram = pg.PlotWidget(background='transparent')
        self.histogram.setMinimumHeight(200)
        self.histogram.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.histogram.setLabel('bottom', 'Select Cell Attribute')
        self.histogram.setLabel('left', 'Probability Density')
        self.histogram.showGrid(x=True, y=True)

        frame_histogram_layout.addLayout(histogram_menu_layout)
        frame_histogram_layout.addWidget(self.histogram)

        self.histogram_menu.dropdownOpened.connect(self._menu_frame_attrs)
        self.histogram_menu.currentTextChanged.connect(self._new_histogram)
        return frame_histogram_widget

    def _get_particle_stat_tab(self):
        particle_plot_widget = QWidget()
        particle_plot_layout = QVBoxLayout(particle_plot_widget)
        particle_stat_menu_layout = QHBoxLayout()
        self.particle_stat_menu = CustomComboBox(self)
        particle_stat_menu_layout.addWidget(self.particle_stat_menu)
        particle_stat_menu_layout.setContentsMargins(40, 0, 0, 0)  # indent the title/menu
        self.particle_stat_menu.addItem('Select Cell Attribute')
        self.particle_stat_plot = pg.PlotWidget(background='transparent')
        self.particle_stat_plot.setLabel('left', 'Select Cell Attribute')
        self.particle_stat_plot.setMinimumHeight(200)
        self.particle_stat_plot.setLabel('bottom', 'Frame')
        self.stat_plot_frame_marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=2))
        self.particle_stat_plot.addItem(self.stat_plot_frame_marker)

        particle_plot_layout.addLayout(particle_stat_menu_layout)
        particle_plot_layout.addWidget(self.particle_stat_plot)

        # connect particle measurements
        self.particle_stat_menu.dropdownOpened.connect(self._menu_frame_attrs)
        self.particle_stat_menu.currentTextChanged.connect(self._plot_particle_statistic)

        return particle_plot_widget

    def _get_time_series_tab(self):
        time_series_widget = QWidget()
        time_series_layout = QVBoxLayout(time_series_widget)
        time_series_menu_layout = QHBoxLayout()
        self.time_series_menu = CustomComboBox(self)
        self.time_series_menu.addItems(['Select Cell Attribute'])
        time_series_menu_layout.addWidget(self.time_series_menu)
        time_series_menu_layout.setContentsMargins(40, 0, 0, 0)  # indent the title/menu

        self.time_series_plot = pg.PlotWidget(background='transparent')
        self.time_series_plot.setLabel('left', 'Select Cell Attribute')
        self.time_series_plot.setLabel('bottom', 'Frame')
        self.time_series_plot.setMinimumHeight(200)
        self.time_series_frame_marker = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('w', width=2))
        self.time_series_plot.addItem(self.time_series_frame_marker)

        time_series_layout.addLayout(time_series_menu_layout)
        time_series_layout.addWidget(self.time_series_plot)

        # connect time series measurements
        self.time_series_menu.dropdownOpened.connect(self._menu_frame_attrs)
        self.time_series_menu.currentTextChanged.connect(self._plot_time_series)
        return time_series_widget

    def _stat_LUT_slider_changed(self):
        self.stat_custom_button.blockSignals(True)
        self.stat_custom_button.setChecked(True)
        self.stat_custom_button.blockSignals(False)
        self._set_stat_LUT_levels(self.stat_LUT_slider.value())

    def _set_stat_LUT_levels(self, levels):
        # TODO: RuntimeWarning: invalid value encountered in cast data=data.astype(int) at level=256, only when working with a stack
        if levels[0] == np.nan and levels[1] == np.nan:
            levels = (0, 1)
        self.canvas.seg_stat_overlay.setLevels(levels)
        self.stat_LUT_slider.blockSignals(True)
        self.stat_LUT_slider.setValue(levels)
        self.stat_LUT_slider.blockSignals(False)
        self.canvas.cb.setLevels(levels)  # TODO: better colorbar tick labels

    def _update_stat_LUT(self):
        if self.stat_frame_button.isChecked():
            self.stat_LUT_type = 'frame'
        elif self.stat_stack_button.isChecked():
            self.stat_LUT_type = 'stack'
        else:
            self.stat_LUT_type = 'custom'

        self._show_seg_overlay()

    def _invert_toggled(self):
        self._apply_overlay_settings()
        self.canvas.img.refresh()

    @property
    def is_inverted(self):
        if hasattr(self, 'left_toolbar'):
            return self.left_toolbar.inverted_checkbox.isChecked()
        else:
            return False

    def _cell_stat_attrs(self, cell):
        """Return all common attributes which are meaningful cell-level metrics"""
        ignored_attrs = {'cycle_stage', 'n', 'frame', 'red', 'green'}
        attrs = cell_scalar_attrs(cell) - ignored_attrs

        return attrs

    def _get_cell_frame_attrs(self, ignored={'frame', 'n', 'green', 'red'}):
        """Return all attributes from any cell in the current frame"""
        if len(self.frame.cells) == 0:
            return []
        keys = set(np.concatenate([dir(cell) for cell in self.frame.cells]))
        # remove __ prefixed attributes
        keys = {item for item in keys if not item.startswith('_')}

        for key in keys.copy():
            # iterate until attribute is found
            for cell in self.frame.cells:
                try:
                    val = getattr(cell, key)
                except AttributeError:
                    continue

                # remove non-scalar attributes
                if not np.isscalar(val):
                    keys.remove(key)
                break
            else:
                keys.remove(key)  # remove if attribute not found in any cell

        return list(keys - ignored)

    def _menu_frame_attrs(self, menu):
        if not self.file_loaded:
            return
        current_attr = menu.currentText()
        menu.blockSignals(True)
        menu.clear()
        keys = self._get_cell_frame_attrs()
        keys = ['Select Cell Attribute'] + natsorted(keys)
        menu.addItems(keys)
        menu.blockSignals(False)
        current_index = menu.findText(current_attr)
        if current_index == -1:
            current_index = 0
        menu.setCurrentIndex(current_index)

    def _get_overlay_attrs(self):
        if not self.file_loaded:
            return
        current_attr = self.seg_overlay_attr.currentText()
        keys = self._get_cell_frame_attrs()
        if hasattr(self.frame, 'heights'):
            keys.append('heights')
        keys = ['Select Cell Attribute'] + natsorted(keys)
        self.seg_overlay_attr.blockSignals(True)
        self.seg_overlay_attr.clear()
        self.seg_overlay_attr.addItems(keys)
        self.seg_overlay_attr.blockSignals(False)
        current_index = self.seg_overlay_attr.findText(current_attr)
        if current_index == -1:
            current_index = 0
        self.seg_overlay_attr.setCurrentIndex(current_index)

    def _new_seg_overlay(self):
        # TODO: adapt LUT range slider to accept floats
        if not self.file_loaded:
            return
        plot_attr = self.seg_overlay_attr.currentText()

        if plot_attr == 'Select Cell Attribute' or plot_attr is None:
            self.stat_LUT_slider.blockSignals(True)
            self.stat_LUT_slider.setRange(0, 255)
            self.stat_LUT_slider.setValue((0, 255))
            self.stat_LUT_slider.blockSignals(False)

        else:
            if plot_attr == 'heights':
                stat = []
                for frame in self.stack.frames:
                    if hasattr(frame, 'heights'):
                        stat.append(frame.heights.flatten())
                stat = np.concatenate(stat)
            else:
                cell_attrs = []
                for frame in self.stack.frames:
                    try:
                        cell_attrs.extend(frame.get_cell_attrs(plot_attr))
                    except AttributeError:
                        continue
                stat = np.array(cell_attrs)
                if len(stat) == 0:
                    print(f'Attribute {plot_attr} not found in cells')
                    return
            LUT_range = (np.nanmin(stat), np.nanmax(stat))
            self.stat_LUT_slider.blockSignals(True)
            self.stat_LUT_slider.setRange(*LUT_range)
            self.stat_LUT_slider.blockSignals(False)

            if self.stat_LUT_type == 'custom':  # change the LUT range to match the new data
                self.stat_frame_button.setChecked(True)

        self._show_seg_overlay()

    def _show_seg_overlay(self, event=None):
        if not self.file_loaded:
            return
        plot_attr = self.seg_overlay_attr.currentText()
        if plot_attr == 'Select Cell Attribute':
            self.canvas.cb.setVisible(False)
            self._clear_seg_stat()
        else:
            self.canvas.cb.setVisible(True)
            if plot_attr == 'heights':
                if not hasattr(self.frame, 'heights'):
                    self.seg_overlay_attr.setCurrentIndex(0)
                    return

                if not hasattr(self.frame, 'z_scale'):
                    print(f'No z scale found for {self.frame.name}, defaulting to 1.')
                    self.left_toolbar.z_size = 1.0

                self._overlay_seg_stat(self.frame.scaled_heights)
            else:
                cell_attrs = np.array(self.frame.get_cell_attrs(plot_attr, fill_value=np.nan))

                value_map = np.concatenate([[np.nan], cell_attrs.astype(float)])
                mask_values = value_map[self.frame.masks]
                self._overlay_seg_stat(mask_values)

    def _overlay_seg_stat(self, stat=None):
        if not self.file_loaded:
            return
        if stat is None:
            stat = self.canvas.seg_stat_overlay.image
        if np.all(np.isnan(stat)):
            levels = (0, 1)
            self.canvas.seg_stat_overlay.clear()
        else:
            stat_range = (np.nanmin(stat), np.nanmax(stat))
            if self.stat_LUT_type == 'frame':
                levels = stat_range
            elif self.stat_LUT_type == 'stack':
                levels = stat_range  # TODO: stack levels
            elif self.stat_LUT_type == 'custom':
                levels = self.stat_LUT_slider.value()
            self.canvas.seg_stat_overlay.setImage(self.canvas.image_transform(stat))

        self.stat_LUT_slider.blockSignals(True)
        self._set_stat_LUT_levels(levels)
        self.stat_LUT_slider.blockSignals(False)

        self.stat_range_labels[0].setText(str(round(levels[0], 2)))
        self.stat_range_labels[1].setText(str(round(levels[1], 2)))

    def _clear_seg_stat(self):
        self.canvas.seg_stat_overlay.clear()

    def _tab_switched(self, index):
        if not self.file_loaded:
            return
        if self.circle_mask is not None:
            self.circle_mask.setVisible(index == 0)
        self._update_tracking_overlay()
        self._FUCCI_overlay()

    @property
    def visual_settings(self):
        return self.left_toolbar._visual_settings

    @visual_settings.setter
    def visual_settings(self, settings):
        self.left_toolbar._visual_settings = settings

    @property
    def masks_visible(self):
        return self.left_toolbar.masks_checkbox.isChecked()

    @masks_visible.setter
    def masks_visible(self, is_visible):
        self.left_toolbar.masks_checkbox.setChecked(is_visible)
        self.canvas.overlay_masks()

    @property
    def outlines_visible(self):
        return self.left_toolbar.outlines_checkbox.isChecked()

    @outlines_visible.setter
    def outlines_visible(self, is_visible):
        self.left_toolbar.outlines_checkbox.setChecked(is_visible)
        self.canvas.overlay_outlines()

    def _update_ROIs_label(self):
        if not self.file_loaded:
            return

        self.left_toolbar.ROIs_label.setText(f'{self.frame.n_cells} ROIs')

    def _cell_prompt_changed(self, cell_n):
        if not self.file_loaded:
            return

        if cell_n == '' or cell_n == 'None':
            cell_n = None
            self._update_tracking_ID_label(None)
            return
        else:
            cell_n = int(cell_n)

        self.select_cell(cell=cell_n)

    def _particle_prompt_changed(self, particle):
        if not self.file_loaded:
            return

        if particle == '' or particle == 'None':
            particle = None
            self._update_cell_label(None)
            return
        else:
            particle = int(particle)

        self.select_cell(particle=particle)

    def _mend_gaps(self):
        if not self.file_loaded:
            return
        if self.left_toolbar.segment_on_stack.isChecked():
            frames = self.stack.frames
        else:
            frames = [self.frame]

        gap_size = self.left_toolbar.gap_size.text()
        if gap_size == '':
            gap_size = None
        else:
            gap_size = int(gap_size)

        self.mend_gaps(frames, gap_size)
        self._update_display()

    def mend_gaps(self, frames, gap_size=None):
        """
        Mend gaps in/between the masks of the specified frames.

        Parameters
        ----------
        frames : list of Frame
            The frames to mend gaps in.
        gap_size : int
            The maximum gap size to mend. If None, the default gap size (based on mean cell area) is used.
        """
        for frame in self._progress_bar(frames):
            mended = frame.mend_gaps(gap_size)
            if mended and hasattr(frame, 'stored_mask_overlay'):
                del frame.stored_mask_overlay
        self.canvas.draw_masks_parallel(frames)

    def _remove_edge_masks(self):
        if not self.file_loaded:
            return
        if self.left_toolbar.segment_on_stack.isChecked():
            frames = self.stack.frames
            current_frame = self.frame_number
        else:
            frames = [self.frame]
            current_frame = 0

        edge_cells = self.remove_edge_masks(frames)

        if self.selected_cell_n in edge_cells[current_frame]:
            # deselect the removed cell if it was selected
            self.select_cell(None)

        self.canvas.draw_outlines()
        self._update_tracking_overlay()
        self._update_display()
        self._update_ROIs_label()

    def remove_edge_masks(self, frames, margin=1):
        """
        Remove masks bordering the edge of the image in the specified frames.

        Parameters
        ----------
        frames : list of Frame
            The frames to remove edge masks from.
        margin : int
            The number of pixels to consider as the edge of the image. Defaults to 1.

        Returns
        -------
        edge_cells : list of list of ints
            The cell IDs which were removed from each frame.
        """
        edge_cells = self.stack.remove_edge_cells(self._progress_bar(frames), margin=margin)
        for deleted_cells, frame in zip(edge_cells, frames):
            print(f'Removed {len(deleted_cells)} edge cells from frame {frame.frame_number}')
            if hasattr(frame, 'stored_mask_overlay'):
                del frame.stored_mask_overlay
        self.canvas.draw_masks_parallel(frames)
        if hasattr(self.stack, 'tracked_centroids'):
            self.left_toolbar.also_save_tracking.setChecked(True)

        return edge_cells

    def _update_cell_diameter(self, diameter):
        self._draw_cell_diameter(diameter)

    def _draw_cell_diameter(self, diameter):
        if self.circle_mask is not None:
            self.canvas.img_plot.removeItem(self.circle_mask)

        if diameter == '':
            return

        diameter = float(diameter)
        padding = 5
        img_shape = self.canvas.img_data.shape[:2]
        self.circle_mask = QGraphicsEllipseItem(padding, img_shape[1] + padding, diameter, diameter)
        self.circle_mask.setBrush(pg.mkBrush(color='#4A90E2'))
        self.canvas.img_plot.addItem(self.circle_mask)

    def _calibrate_diameter(self):
        channels = self.left_toolbar.segmentation_channels
        diam = self.calibrate_cell_diameter(self.frame.img, channels)

        print(f'Computed cell diameter {diam:.2f} with channels {channels}')
        self.left_toolbar.cell_diameter.setText(f'{diam:.2f}')

    def calibrate_cell_diameter(self, img, channels):
        """
        Automatically calibrate the cell diameter for the specified image using Cellpose.

        Parameters
        ----------
        img : np.ndarray
            The image to calibrate the cell diameter for.
        channels : length-2 tuple of ints
            The channels (membrane, nuclei) to use for segmentation.
            0 is grayscale, 1-3 are RGB, 4 is FUCCI (combined red and green) and is only available for nuclei.
        """
        if not self.file_loaded:
            return

        if not hasattr(self, 'size_model'):
            from cellpose import models

            model_type = 'cyto3'
            self.cellpose_model = models.CellposeModel(gpu=True, model_type=model_type)
            self.size_model_path = models.size_model_path(model_type)
            self.size_model = models.SizeModel(self.cellpose_model, pretrained_size=self.size_model_path)

        if channels[1] == 4:  # FUCCI channel
            from segmentation_tools.image_segmentation import combine_FUCCI_channels

            if channels[0] == 0:
                membrane = np.mean(img, axis=-1)  # grayscale
            else:
                membrane = img[..., channels[0] - 1]  # fetch specified channel

            nuclei = combine_FUCCI_channels(img)[..., 0]
            img = np.stack([nuclei, membrane], axis=-1)
            diam, _ = self.size_model.eval(img, channels=[2, 1])
        else:
            diam, _ = self.size_model.eval(img, channels=channels)

        return diam

    def _segment_frame(self):
        if not self.file_loaded:
            return

        diameter = self.left_toolbar.cell_diameter.text()
        if diameter == '':
            diameter = None
        else:
            diameter = float(diameter)
        channels = self.left_toolbar.segmentation_channels

        self.segment([self.frame], diameter=diameter, channels=channels)

        # update the display
        self.left_toolbar.cell_diameter.setText(f'{self.frame.cell_diameter:.2f}')
        self.masks_visible = True
        self.canvas.draw_masks()
        self._update_display()
        self._FUCCI_overlay()

    def _segment_stack(self):
        if not self.file_loaded:
            return

        if self.is_zstack:  # segment the zstack
            for frame in self.stack.frames:
                frame.img = frame.zstack[self.zstack_number]

        diameter = self.left_toolbar.cell_diameter.text()
        if diameter == '':
            diameter = None
        else:
            diameter = float(diameter)
        channels = self.left_toolbar.segmentation_channels

        self.segment(self.stack.frames, diameter=diameter, channels=channels)

        # update the display
        self.left_toolbar.cell_diameter.setText(f'{self.frame.cell_diameter:.2f}')
        self.masks_visible = True
        self.canvas.draw_masks()
        self._update_display()
        self._FUCCI_overlay()

    def segment(self, frames, diameter=None, channels=[0, 0]):
        """
        Performs segmentation on the specified frames using Cellpose.

        Parameters
        ----------
        frames : list of frames
            The frames to segment.
        diameter : float
            The cell diameter to use for segmentation. If None, the diameter is automatically calibrated.
        channels : length-2 tuple of ints
            The channels (membrane, nuclei) to use for segmentation.
            0 is grayscale, 1-3 are RGB, 4 is FUCCI (combined red and green) and is only available for nuclei.
        """

        if not hasattr(self, 'cellpose_model'):
            from cellpose import models

            model_type = 'cyto3'
            self.cellpose_model = models.CellposeModel(gpu=True, model_type=model_type)

        for frame in self._progress_bar(frames, desc='Segmenting frames'):
            if diameter is None:
                diameter = self.calibrate_cell_diameter(frame.img, channels)
            frame.cell_diameter = diameter
            if channels[1] == 4:  # FUCCI channel
                img = frame.img.copy()
                from segmentation_tools.image_segmentation import combine_FUCCI_channels

                if channels[0] == 0:
                    membrane = np.mean(img, axis=-1)  # grayscale
                else:
                    membrane = img[..., channels[0] - 1]  # fetch specified channel

                nuclei = combine_FUCCI_channels(img)[..., 0]
                img = np.stack([nuclei, membrane], axis=-1)
                masks, _, _ = self.cellpose_model.eval(img, channels=[2, 1], diameter=diameter)
            else:
                masks, _, _ = self.cellpose_model.eval(frame.img, channels=channels, diameter=diameter)
            frame.masks = masks
            self.replace_segmentation(frame)

            if frame == self.frame:
                self._update_ROIs_label()

    def clear_masks(self):
        """
        Clear the segmentation masks for the current frame.
        """

        if not self.file_loaded:
            return

        if self.left_toolbar.segment_on_stack.isChecked():
            frames = self.stack.frames
        else:
            frames = [self.frame]

        for frame in frames:
            frame.masks = np.zeros_like(frame.masks)
            self.replace_segmentation(frame)
            if hasattr(self.stack, 'tracked_centroids'):
                t = self.stack.tracked_centroids
                self.stack.tracked_centroids = t[t.frame != frame.frame_number]
                self.left_toolbar.also_save_tracking.setChecked(True)

        self._update_display()

    def _progress_bar(self, iterable, desc=None, length=None):
        if length is None:
            length = len(iterable)

        if length <= 1:
            return iterable
        else:
            # Initialize tqdm progress bar
            tqdm_bar = tqdm(iterable, desc=desc, total=length)

            # Initialize QProgressBar
            qprogress_bar = QProgressBar()
            qprogress_bar.setMaximum(length)

            # Set size policy to match the status bar width
            qprogress_bar.setFixedHeight(int(self.statusBar().height() * 0.8))
            qprogress_bar.setFixedWidth(int(self.statusBar().width() * 0.2))

            # Temporarily hide existing permanent status bar widgets
            self.status_coordinates.setVisible(False)
            self.status_pixel_value.setVisible(False)

            self.statusBar().addPermanentWidget(qprogress_bar)

            # Custom iterator to update both progress bars
            def custom_iterator():
                self.is_iterating = True
                self.cancel_iter = False
                for i, item in enumerate(iterable):
                    QApplication.processEvents()  # allow updates, check for key presses
                    if self.cancel_iter:
                        self.is_iterating = False
                        QMessageBox.warning(self, 'Operation Cancelled', 'Operation cancelled by user.')
                        break
                    yield item
                    tqdm_bar.update(1)
                    qprogress_bar.setValue(i + 1)
                tqdm_bar.close()
                self.statusBar().removeWidget(qprogress_bar)
                # Restore existing permanent status bar widgets
                self.status_coordinates.setVisible(True)
                self.status_pixel_value.setVisible(True)
                self.is_iterating = False

            return custom_iterator()

    def replace_segmentation(self, frame):
        """
        Replace frame cells and outlines using frame.masks as reference.
        Called after any re-segmentation operation.
        """

        frame.has_outlines = False
        frame.outlines = utils.masks_to_outlines(frame.masks)
        frame.n_cells = np.max(frame.masks)
        frame.cells = np.array(
            [Cell(n, np.empty((0, 2)), frame_number=frame.frame_number, parent=frame) for n in range(frame.n_cells)]
        )
        if hasattr(frame, 'stored_mask_overlay'):
            del frame.stored_mask_overlay
            self.canvas.draw_masks_bg(frame)

    def get_tracked_FUCCI(self):
        """
        Get cell cycle classification by tracking particles through time.
        Tries to identify G1-S and S-G2 transitions by characteristic changes in red/green intensity.
        """
        if not self.file_loaded:
            return
        if not hasattr(self.stack, 'tracked_centroids'):
            self.statusBar().showMessage('No tracked centroids found.', 2000)
            return

        self.stack.measure_FUCCI_by_transitions(progress=self._progress_bar)

        for frame in self.stack.frames:
            self._get_red_green(frame)

        self._FUCCI_overlay()

    def cell_red_green_intensities(self, event=None, percentile=90, sigma=4):
        """
        Measure red and green channel intensities for each cell in the current frame.

        Parameters
        ----------
        percentile : float
            The percentile to use for thresholding the red and green channels.
        sigma:
            The sigma value to use for Gaussian smoothing of the red and green channels.
        """

        if not self.file_loaded:
            return
        for frame in self._progress_bar(self.stack.frames):
            frame.get_red_green_intensities(percentile, sigma)

    def _propagate_FUCCI_toggled(self, state):
        """Propagate the FUCCI labels forward in time."""
        if state != 2 or not self.file_loaded:
            return
        if not hasattr(self.stack, 'tracked_centroids'):
            self.statusBar().showMessage('No tracked centroids found.', 2000)
            return

        self._convert_red_green()
        self.stack.propagate_FUCCI_labels(progress=self._progress_bar)
        for frame in self.stack.frames:
            self._get_red_green(frame)
        self._FUCCI_overlay()

    def _clear_FUCCI_frame(self):
        if not self.file_loaded:
            return
        self.clear_FUCCI([self.frame])

    def _clear_FUCCI_stack(self):
        if not self.file_loaded:
            return
        self.clear_FUCCI(self.stack.frames)

    def clear_FUCCI(self, frames):
        """
        Clear the FUCCI labels for the specified frames.
        """
        for frame in frames:
            frame.set_cell_attrs(['red', 'green'], np.array([[False, False] for _ in range(frame.n_cells)]).T)
        self._FUCCI_overlay()

    def _measure_FUCCI_frame(self):
        if not self.file_loaded:
            return
        red_threshold, green_threshold, percent_threshold = (
            self.left_toolbar.red_threshold,
            self.left_toolbar.green_threshold,
            self.left_toolbar.percent_threshold,
        )
        self.measure_FUCCI(
            [self.frame], red_threshold=red_threshold, green_threshold=green_threshold, percent_threshold=percent_threshold
        )

    def _measure_FUCCI_stack(self):
        if not self.file_loaded:
            return
        red_threshold, green_threshold, percent_threshold = (
            self.left_toolbar.red_threshold,
            self.left_toolbar.green_threshold,
            self.left_toolbar.percent_threshold,
        )
        self.measure_FUCCI(
            self.stack.frames, red_threshold=red_threshold, green_threshold=green_threshold, percent_threshold=percent_threshold
        )

    def measure_FUCCI(self, frames, red_threshold=None, green_threshold=None, orange_brightness=1, percent_threshold=0.15):
        """
        Measure the FUCCI labels for the specified frames by snapshot thresholding.

        Parameters
        ----------
        frames : list of Frame
            The frames to measure FUCCI for.
        red_threshold : float
            The threshold to use for the red channel. If None, the threshold is determined automatically (but may be inaccurate).
        green_threshold : float
            The threshold to use for the green channel. If None, the threshold is determined automatically (but may be inaccurate).
        orange_brightness : float
            The relative brightness of red compared to green to be considered orange. Defaults to 1.
            If many G1 cells are being classified as G2, consider increasing this value.
        percent_threshold : float
            The fractional area of the cell which must be above the red/green threshold to be considered positive.
        """
        for frame in self._progress_bar(frames, desc='Measuring FUCCI'):
            if self.is_zstack:
                img = frame.zstack[self.zstack_number]
            else:
                img = frame.img
            if not hasattr(frame, 'FUCCI'):
                frame.FUCCI = img[..., 0], img[..., 1]  # use the red and green channels

            frame.measure_FUCCI(
                red_fluor_threshold=red_threshold,
                green_fluor_threshold=green_threshold,
                orange_brightness=orange_brightness,
                percent_threshold=percent_threshold,
            )
            self._get_red_green(frame)

        self.red_threshold = self.frame.red_fluor_threshold
        self.green_threshold = self.frame.green_fluor_threshold
        self.FUCCI_dropdown = 3
        self._FUCCI_overlay()

    def delete_particle_head(self):
        """
        Delete all masks preceding (and including) the current frame for the selected particle.
        """
        if not self.file_loaded:
            return
        if not hasattr(self.stack, 'tracked_centroids'):
            self.delete_cell_mask(self.selected_cell_n)
            return

        else:
            particle_n = self.selected_particle_n
            current_frame_n = self.frame_number
            t = self.stack.tracked_centroids

            head_cell_numbers, head_frame_numbers = np.array(
                t[(t.particle == particle_n) & (t.frame <= current_frame_n)][['cell_number', 'frame']]
            ).T
            self.left_toolbar.also_save_tracking.setChecked(True)
            for cell_n, frame_n in zip(head_cell_numbers, head_frame_numbers):
                frame = self.stack.frames[frame_n]
                if hasattr(frame, 'stored_mask_overlay'):
                    self.canvas.add_cell_highlight(cell_n, frame, color='none', layer='mask')
                self.delete_cell_mask(cell_n, frame)

            # reselect the particle
            self.select_cell(particle=particle_n)
            self._plot_particle_statistic()

    def delete_particle_tail(self):
        """
        Delete all masks following (and including) the current frame for the selected particle.
        """
        if not self.file_loaded:
            return
        if not hasattr(self.stack, 'tracked_centroids'):
            self.delete_cell_mask(self.selected_cell_n)
            return

        else:
            particle_n = self.selected_particle_n
            current_frame_n = self.frame_number
            t = self.stack.tracked_centroids

            head_cell_numbers, head_frame_numbers = np.array(
                t[(t.particle == particle_n) & (t.frame >= current_frame_n)][['cell_number', 'frame']]
            ).T
            self.left_toolbar.also_save_tracking.setChecked(True)
            for cell_n, frame_n in zip(head_cell_numbers, head_frame_numbers):
                frame = self.stack.frames[frame_n]
                if hasattr(frame, 'stored_mask_overlay'):
                    self.canvas.add_cell_highlight(cell_n, frame, color='none', layer='mask')
                self.delete_cell_mask(cell_n, frame)

            # reselect the particle
            self.select_cell(particle=particle_n)
            self._plot_particle_statistic()

    def delete_particle(self, event=None, particle_n=None):
        """
        Delete all masks for the selected particle.

        Parameters
        ----------
        particle_n : int
            The particle number to delete masks for. If None, the selected particle is used.
        """
        if not self.file_loaded:
            return
        if not hasattr(self.stack, 'tracked_centroids'):
            self.delete_cell_mask(self.selected_cell_n)
            return

        else:
            if particle_n is None:
                particle_n = self.selected_particle_n
            t = self.stack.tracked_centroids
            self.left_toolbar.also_save_tracking.setChecked(True)

            cell_numbers, frame_numbers = np.array(t[t.particle == particle_n][['cell_number', 'frame']]).T
            for cell_n, frame_n in zip(cell_numbers, frame_numbers):
                frame = self.stack.frames[frame_n]
                if hasattr(frame, 'stored_mask_overlay'):
                    self.canvas.add_cell_highlight(cell_n, frame, color='none', layer='mask')
                self.delete_cell_mask(cell_n, frame)
            self._update_tracking_overlay()

    def clear_tracking(self):
        """
        Clear all tracking data for the stack.
        """
        if not self.file_loaded:
            return
        if not hasattr(self.stack, 'tracked_centroids'):
            return

        del self.stack.tracked_centroids
        self.canvas.clear_overlay('tracking')
        self._clear_particle_statistic()
        self._random_recolor()  # recolor masks to signify unlinking

    def _random_recolor(self):
        if not self.file_loaded:
            return
        for frame in self.stack.frames:
            if hasattr(frame, 'stored_mask_overlay'):
                del frame.stored_mask_overlay
            for cell in frame.cells:
                del cell.color_ID
        self.canvas.draw_masks_parallel()

        self.canvas.draw_masks()

    def _delete_mitosis(self):
        if not self.file_loaded:
            return
        if not hasattr(self.stack, 'mitoses'):
            return
        candidate_particle = self.selected_particle_n
        if candidate_particle is None:
            return
        idx, col = np.where(self.stack.mitoses[['mother', 'daughter1', 'daughter2']] == candidate_particle)
        if len(idx) == 0:  # not in mitoses
            return
        elif len(idx) == 1:
            idx = idx[0]
        else:
            idx = idx[np.argmin(col)]

        deleted_m = self.stack.mitoses.iloc[[idx]][['frame', 'mother', 'daughter1', 'daughter2']].to_string(index=False)
        self.stack.mitoses.drop(self.stack.mitoses.index[idx], inplace=True)
        print(f'Deleted mitosis\n{deleted_m}')
        self._update_tracking_overlay()

    def _add_mitosis(self, event=None):
        print('Add mitosis')
        if not self.file_loaded:
            return
        if not hasattr(self.stack, 'tracked_centroids'):
            self.statusBar().showMessage("Can't create mitoses without tracking data.", 2000)
            return

        if not self.mitosis_mode:
            self._start_mitosis(frame_number=self.frame_number + 1)  # assume pressed on last frame of mother cell
        else:
            self._cancel_mitosis()

    def _start_mitosis(self, frame_number=None):
        if frame_number is None:
            frame_number = self.frame_number
        self.mitosis_mode = 1
        self.current_mitosis = {'frame': frame_number}
        print(f'Starting mitosis at frame {frame_number}. Select mother cell.')

    def _mitosis_selected(self, particle_n):
        if not hasattr(self, 'current_mitosis'):
            return
        keys = ['mother', 'daughter1', 'daughter2']
        key = keys[self.mitosis_mode - 1]
        self.current_mitosis[key] = particle_n
        if self.mitosis_mode == 3:
            self._end_mitosis()
            return

        if particle_n is not None:
            if key.startswith('mother'):
                color = 'red'
            else:
                color = 'lime'
            self.canvas.add_cell_highlight(
                self.cell_from_particle(particle_n), color=color, layer='mitosis', alpha=self.canvas.masks_alpha
            )
        print(f'Selected {key} cell: {particle_n}. Select {keys[self.mitosis_mode]} cell.')
        self.mitosis_mode += 1

    def _cancel_mitosis(self):
        if not hasattr(self, 'current_mitosis'):
            return
        self.mitosis_mode = 0
        del self.current_mitosis
        print('Add mitosis cancelled.')
        self.canvas.clear_overlay('mitosis')

    def _end_mitosis(self):
        if not hasattr(self, 'current_mitosis'):
            return

        self.stack.add_mitosis(self.current_mitosis)
        print(f'Added mitosis {self.current_mitosis}')
        self.canvas.clear_overlay('mitosis')
        self.mitosis_mode = 0
        del self.current_mitosis
        self._update_tracking_overlay()

    def _get_mitoses(self):
        if not self.file_loaded:
            return
        distance_threshold, score_cutoff, weights = self.left_toolbar.mitosis_params
        self.get_mitoses(distance_threshold=distance_threshold, score_cutoff=score_cutoff, weights=weights)
        self._update_tracking_overlay()

    def get_mitoses(self, event=None, **kwargs):
        """
        Identify mitoses for the stack using the specified parameters.
        Scoring and identification has been manually tuned for confluent MDCK monolayers.

        Parameters
        ----------
        distance_threshold : float
            The maximum distance between daughter centroids for a mitosis to be considered valid.
        score_cutoff : float
            The minimum score for a mitosis to be considered valid. Defaults to 1.
        weights : list of five floats
            The weights to use for the mitosis scoring function. Defaults to [1, 1, 1, 1, 1, 1].
            Weights correspond to the following features:
            [mother_circularity, daughter_circularities, daughter_centroid_asymmetry, daughter_angle, center_of_mass_displacement]
        """

        self.stack.get_mitoses(**kwargs)

    def _apply_new_weights(self):
        if not self.file_loaded:
            return
        distance_threshold, score_cutoff, weights = self.left_toolbar.mitosis_params
        if not hasattr(self.stack, 'mitosis_scores'):
            # get mitosis scores if they don't already exist
            self.get_mitoses(distance_threshold=distance_threshold, score_cutoff=score_cutoff, weights=weights)
        else:
            # recompute mitosis scores with new weights
            from segmentation_tools.mitosis_detection import get_mitosis_scores, threshold_mitoses

            self.stack.mitosis_scores = get_mitosis_scores(self.stack.mitosis_scores, weights=weights)
            mitoses = threshold_mitoses(self.stack.mitosis_scores, threshold=score_cutoff)
            self.stack.mitoses = mitoses.astype(
                {'mother': 'Int64', 'daughter1': 'Int64', 'daughter2': 'Int64'}
            )  # nullable integer columns

        self._update_tracking_overlay()

    def _update_tracking_overlay(self):
        sender = self.sender()
        if hasattr(self.stack, 'tracked_centroids'):
            self.left_toolbar.update_tracking_label(self.stack.tracked_centroids['particle'].nunique())
        if hasattr(self.stack, 'mitoses'):
            self.left_toolbar.update_mitoses_label(len(self.stack.mitoses))
        # enforce checkbox exclusivity
        if sender == self.left_toolbar.highlight_track_ends_button and sender.isChecked():
            self.left_toolbar.highlight_mitoses_button.setChecked(False)
        elif sender == self.left_toolbar.highlight_mitoses_button and sender.isChecked():
            self.left_toolbar.highlight_track_ends_button.setChecked(False)

        if not self.file_loaded or self.left_toolbar.tabbed_widget.currentIndex() != 2:
            self.canvas.clear_overlay('tracking')
            return
        else:
            if self.left_toolbar.highlight_track_ends_button.isChecked():
                self._highlight_track_ends()
            elif self.left_toolbar.highlight_mitoses_button.isChecked():
                self._highlight_mitoses()
            else:
                self.canvas.clear_overlay('tracking')
            return

    def _highlight_mitoses(self):
        if not hasattr(self.stack, 'mitoses'):
            return

        # get all mitoses within n frames of the current frame
        tail_length = 5

        mitoses = self.stack.mitoses[abs(self.stack.mitoses.frame - self.frame_number) <= tail_length]

        self.canvas.clear_overlay('tracking')
        if len(mitoses) == 0:
            return

        def highlight_timepoint(cell, color, t1, t2):
            dt = t2 - t1
            if dt >= 0:
                alpha = 1 - (dt + 1) / (tail_length + 1)
                self.canvas.add_cell_highlight(
                    cell, color=color, alpha=alpha, layer='tracking', img_type='outlines', seg_alpha=True, mode='blend'
                )

        for _, m in mitoses.iterrows():
            mitosis_frame = m['frame']
            mother, daughter1, daughter2 = (self.cell_from_particle(m[p]) for p in ['mother', 'daughter1', 'daughter2'])
            if mother is not None:
                highlight_timepoint(mother, 'red', self.frame_number, mitosis_frame)
            if daughter1 is not None:
                highlight_timepoint(daughter1, 'lime', mitosis_frame, self.frame_number)
            if daughter2 is not None:
                highlight_timepoint(daughter2, 'lime', mitosis_frame, self.frame_number)
        return

    def _highlight_track_ends(self):
        if hasattr(self.stack, 'tracked_centroids'):
            # get the start and end points of each track
            t = self.stack.tracked_centroids
            track_ends = t.groupby('particle').agg({'frame': ['first', 'last']})
            track_ends.columns = ['start', 'end']
            track_ends = track_ends.reset_index()

            births = track_ends[track_ends.start == self.frame_number]['particle']
            deaths = track_ends[track_ends.end == self.frame_number]['particle']
            if self.frame_number == 0:
                births = []
            elif self.frame_number == len(self.stack.frames) - 1:
                deaths = []

            birth_cells = t[(t.frame == self.frame_number) & t.particle.isin(births)]['cell_number']
            death_cells = t[(t.frame == self.frame_number) & t.particle.isin(deaths)]['cell_number']
            both = np.intersect1d(birth_cells, death_cells)
            colors = (
                ['lime'] * len(birth_cells) + ['red'] * len(death_cells) + ['orange'] * len(both)
            )  # TODO: pick better colors which don't overlap with FUCCI
            self.canvas.highlight_cells(
                np.concatenate([birth_cells, death_cells, both]),
                alpha=0.5,
                cell_colors=colors,
                layer='tracking',
                img_type='outlines',
            )

        else:
            self.canvas.clear_overlay('tracking')

    def _canvas_wheelEvent(self, event):
        if not self.file_loaded:
            return
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:  # shift+scroll = z-stack
            if self.is_zstack:
                if event.angleDelta().y() > 0:  # scroll up = higher in z-stack
                    self.zstack_slider.setValue(min(self.zstack_slider.value() + 1, len(self.frame.zstack) - 1))
                else:
                    self.zstack_slider.setValue(max(self.zstack_slider.value() - 1, 0))
        else:  # scroll = frame
            if event.angleDelta().y() < 0:  # scroll down = next frame
                self.change_current_frame(min(self.frame_number + 1, len(self.stack.frames) - 1))
            else:  # scroll up = previous frame
                self.change_current_frame(max(self.frame_number - 1, 0))

    def update_zstack_number(self, zstack_number):
        """
        Move in z to the specified z-stack slice.
        """
        if not self.file_loaded or not self.is_zstack:
            return
        self.zstack_number = zstack_number
        self.frame.img = self.frame.zstack[self.zstack_number]
        self._update_coordinate_label()
        self.imshow()

    def _refresh_right_toolbar(self, cell_attr=None):
        if cell_attr is None:
            self._plot_histogram()
            self._show_seg_overlay()
            self._plot_particle_statistic()
        else:
            if self.histogram_menu.currentText() == cell_attr:
                self._plot_histogram()
            if self.seg_overlay_attr.currentText() == cell_attr:
                self._show_seg_overlay()
            if self.particle_stat_menu.currentText() == cell_attr:
                self._plot_particle_statistic()

    def _measure_volumes(self):
        if not self.file_loaded:
            return
        if self.left_toolbar.volumes_on_stack.isChecked():
            frames = self.stack.frames
        else:
            frames = [self.frame]

        self.measure_volumes(frames)

        # update the display if necessary
        self._refresh_right_toolbar('volume')

    def measure_volumes(self, frames):
        """
        Measure the volumes of the specified frames.
        This is accomplished by integrating heights over each cell mask.

        Parameters
        ----------
        frames : list of Frame
            The frames to measure volumes for.
        """
        for frame in self._progress_bar(frames):
            if not hasattr(frame, 'heights'):
                if hasattr(frame, 'zstack'):
                    peak_prominence = self.left_toolbar.peak_prominence.text()
                    coverslip_height = self.left_toolbar.coverslip_height.text()
                    if peak_prominence == '':
                        peak_prominence = 0.01
                    else:
                        peak_prominence = float(peak_prominence)
                    if coverslip_height == '':
                        coverslip_height = self.calibrate_coverslip_height([frame])
                        self.left_toolbar.coverslip_height.setText(f'{coverslip_height:.2f}')
                    else:
                        coverslip_height = float(coverslip_height)
                    self.measure_heights([frame], peak_prominence, coverslip_height)
                else:
                    raise ValueError(f'No heights or z-stack available to measure volumes for {frame.name}.')

            if not hasattr(frame, 'z_scale'):
                print(f'No z-scale available for {frame.name}. Defaulting to 1.')
                self.left_toolbar.z_size = 1.0
            if not hasattr(frame, 'scale'):
                print(f'No scale available for {frame.name}. Defaulting to 0.1625.')
                self.left_toolbar.xy_size = 0.1625
                frame.scale = 0.1625  # 40x objective with 0.325 µm/pixel camera
            frame.get_volumes()

    def _calibrate_coverslip_height(self):
        if not self.file_loaded:
            return
        if self.left_toolbar.volumes_on_stack.isChecked():
            frames = self.stack.frames
            if not all(hasattr(frame, 'zstack') for frame in frames):
                raise ValueError('No z-stacks available to calibrate coverslip height.')
            if len(np.unique([frame.zstack.shape[0] for frame in frames])) > 1:
                raise ValueError('Z-stack lengths are not consistent.')
        else:
            frames = [self.frame]

        coverslip_height = self.calibrate_coverslip_height(frames)
        self.left_toolbar.coverslip_height.setText(f'{coverslip_height:.2f}')

    def calibrate_coverslip_height(self, frames):
        """
        Identify the bottom of the sample from the specified frames.

        Parameters
        ----------
        frames : list of Frame
            The frames to calibrate the coverslip height for.

        Returns
        -------
        coverslip_height : float
            The height of the coverslip (in microns) above the bottom of the z-stack.
        """
        from segmentation_tools.heightmap import get_coverslip_z

        z_profile = []
        for z_index in range(frames[0].zstack.shape[0]):
            if self.is_grayscale:
                z_profile.append(np.mean(np.concatenate([frame.zstack[z_index].flatten() for frame in frames])))
            else:
                z_profile.append(np.mean(np.concatenate([frame.zstack[z_index, ..., 2].flatten() for frame in frames])))

        if not hasattr(self.frame, 'z_scale'):
            print(f'No z-scale available for {self.frame.name}. Defaulting to 1.')
            self.left_toolbar.z_size = 1.0
        scale = self.frame.z_scale

        coverslip_height = get_coverslip_z(z_profile, scale=scale, precision=0.01)
        for frame in frames:
            frame.coverslip_height = coverslip_height
        return coverslip_height

    def _measure_heights(self):
        if not self.file_loaded:
            return
        if self.left_toolbar.volumes_on_stack.isChecked():
            frames = self.stack.frames
        else:
            frames = [self.frame]

        peak_prominence = self.left_toolbar.peak_prominence.text()
        if peak_prominence == '':
            peak_prominence = 0.01
        else:
            peak_prominence = float(peak_prominence)

        coverslip_height = self.left_toolbar.coverslip_height.text()
        if coverslip_height == '':
            self._calibrate_coverslip_height()
            coverslip_height = self.left_toolbar.coverslip_height.text()
        coverslip_height = float(coverslip_height)

        self.measure_heights(frames, peak_prominence, coverslip_height)
        self._show_seg_overlay()
        self.left_toolbar.volume_button.setEnabled(True)

    def measure_heights(self, frames, peak_prominence=0.01, coverslip_height=None):
        """
        Compute the heightmap of the monolayer for the specified frames.

        Parameters
        ----------
        frames : list of Frame
            The frames to measure heights for.
        peak_prominence : float
            The minimum prominence of a peak to be considered the top of a cell. Defaults to 0.01.
            If the height map is very noisy, consider increasing this value. If cell membranes are not being detected, consider decreasing it.
        coverslip_height : float
            The height of the coverslip above the bottom of the z-stack. If None, the bottom of the z-stack is treated as the coverslip.
        """
        from segmentation_tools.heightmap import get_heights

        for frame in self._progress_bar(frames):
            if not hasattr(frame, 'zstack'):
                raise ValueError(f'No z-stack available to measure heights for {frame.name}.')
            else:
                if self.is_grayscale:
                    membrane = frame.zstack
                else:
                    membrane = frame.zstack[..., 2]  # TODO: allow user to specify membrane channel
                frame.heights = get_heights(membrane, peak_prominence=peak_prominence)
                frame.to_heightmap()
                if coverslip_height:
                    frame.coverslip_height = coverslip_height

    def _compute_spherical_volumes(self):
        if not self.file_loaded:
            return

        if self.left_toolbar.volumes_on_stack.isChecked():
            frames = self.stack.frames
        else:
            frames = [self.frame]

        self.get_spherical_volumes(frames)

        self._refresh_right_toolbar('volume')

    def get_spherical_volumes(self, frames):
        """
        Compute the volumes of cells if their areas are treated as circular cross-sections of spheres.
        This is used for estimating the volumes of cells in suspension.

        Parameters
        ----------
        frames : list of Frame
            The frames to compute spherical volumes for
        """
        for frame in self._progress_bar(frames):
            frame.get_spherical_volumes()

    def change_current_frame(self, frame_number, reset=False):
        """
        Change the current frame to the specified frame number.

        Parameters
        ----------
        frame_number : int
            The frame number to change to.
        reset : bool
            Whether to reset the display to the default view.
        """
        if not self.file_loaded:
            return

        start_time = time.time()
        self.frame_number = frame_number
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_number)
        self.frame_slider.blockSignals(False)
        self.frame = self.stack.frames[self.frame_number]
        self.globals_dict['frame'] = self.frame

        if hasattr(self.frame, 'zstack'):
            self.frame.img = self.frame.zstack[self.zstack_number]
            self.zstack_slider.setVisible(True)
            self.zstack_slider.setRange(0, self.frame.zstack.shape[0] - 1)
            self.is_zstack = True
        else:
            self.frame.img = self.frame.img
            self.zstack_slider.setVisible(False)
            self.is_zstack = False

        if self.is_zstack or hasattr(self.frame, 'heights'):
            self.left_toolbar.volume_button.setEnabled(True)
            if not self.is_zstack:  # enable/disable z-stack specific options
                self.left_toolbar.get_heights_button.setEnabled(False)
                self.left_toolbar.peak_prominence.setEnabled(False)
            else:
                self.left_toolbar.get_heights_button.setEnabled(True)
                self.left_toolbar.peak_prominence.setEnabled(True)

            if hasattr(self.frame, 'coverslip_height'):
                self.left_toolbar.coverslip_height.setText(f'{self.frame.coverslip_height:.2f}')
            else:
                self.left_toolbar.coverslip_height.setText('')
        else:
            self.left_toolbar.get_heights_button.setEnabled(False)
            self.left_toolbar.peak_prominence.setEnabled(False)
            self.left_toolbar.volume_button.setEnabled(False)
            self.left_toolbar.coverslip_height.setText('')

        self.imshow()

        if reset:
            self.reset_display()
        else:
            # preserve selected cell if tracking info is available
            if hasattr(self, 'selected_particle') and self.selected_particle_n is not None:
                self.select_cell(particle=self.selected_particle_n)

            # or clear highlight
            else:
                self.canvas.clear_overlay('selection')  # no tracking data, clear highlights

        if len(self.frame.cells) > 0 and not hasattr(self.frame.cells[0], 'green'):
            self._get_red_green()

        if hasattr(self.frame, 'red_fluor_threshold'):
            self.left_toolbar.red_threshold = self.frame.red_fluor_threshold
            self.left_toolbar.green_threshold = self.frame.green_fluor_threshold
        else:
            self.left_toolbar.red_threshold = None
            self.left_toolbar.green_threshold = None

        self._update_voxel_size_labels()

        if hasattr(self.frame, 'cell_diameter'):
            self.left_toolbar.cell_diameter.setText(f'{self.frame.cell_diameter:.2f}')

        if self.FUCCI_dropdown != 0:
            self._FUCCI_overlay()

        # frame marker on stat plot
        self.stat_plot_frame_marker.setPos(self.frame_number)
        self.time_series_frame_marker.setPos(self.frame_number)
        self.status_frame_number.setText(f'Frame: {frame_number}')

        if debug_execution_times:
            print(f'Total frame change time: {time.time() - start_time:.3f}s')

    @property
    def FUCCI_dropdown(self):
        return self.left_toolbar.FUCCI_dropdown.currentIndex()

    @FUCCI_dropdown.setter
    def FUCCI_dropdown(self, index):
        self.left_toolbar.FUCCI_dropdown.setCurrentIndex(index)

    def _update_coordinate_label(self, x=None, y=None):
        """Update the status bar with the current cursor coordinates."""
        if x is None or y is None:
            x, y = self.canvas.cursor_pixels

        coordinates = f'{x}, {y}'
        if hasattr(self, 'zstack_number'):
            coordinates += f', {self.zstack_number}'
        self.status_coordinates.setText(f'Coordinates: ({coordinates})')
        pixel_value = self._get_pixel_value(x, y)
        if self.is_grayscale:
            pixel_string = f'Gray: {pixel_value[0]}'
        else:
            pixel_string = ', '.join(f'{color}: {str(p)}' for color, p in zip(('R', 'G', 'B'), pixel_value))
        self.status_pixel_value.setText(pixel_string)

    def _get_pixel_value(self, x, y):
        """Get the pixel value at the current cursor position."""
        if not self.file_loaded:
            return None, None, None
        elif not hasattr(self, 'frame'):  # catch case where file_loaded is mistakenly True due to some exception
            return None, None, None
        img = self.canvas.inverse_image_transform(self.canvas.img_data)

        if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:  # outside image bounds
            return None, None, None

        if self.is_grayscale:
            return [img[y, x]]

        hidden_channels = np.where(~np.array(self.left_toolbar.RGB_visible))[0]
        pixel_value = list(img[y, x])
        for channel in hidden_channels:
            pixel_value[channel] = None
        return pixel_value

    def _update_cell_label(self, cell_n):
        """Update the status bar with the selected cell number."""
        if cell_n is None:
            self.status_cell.setText('Selected Cell: None')
            self.selected_cell_prompt.setText('')
        else:
            self.status_cell.setText(f'Selected Cell: {cell_n}')
            self.selected_cell_prompt.setText(str(cell_n))

    def _update_tracking_ID_label(self, tracking_ID):
        """Update the status bar with the current tracking ID."""
        if tracking_ID is None:
            self.status_tracking_ID.setText('Tracking ID: None')
            self.selected_particle_prompt.setText('')
        else:
            self.status_tracking_ID.setText(f'Tracking ID: {tracking_ID}')
            self.selected_particle_prompt.setText(str(tracking_ID))

    def _track_centroids(self):
        if not self.file_loaded:
            return

        tracking_range = self.left_toolbar.tracking_range.text()
        if tracking_range == '':
            tracking_range = None
        else:
            tracking_range = float(tracking_range)
        memory = self.left_toolbar.memory_range.text()
        if memory == '':
            memory = 0

        self.statusBar().showMessage('Tracking centroids...')

        try:
            self.track_centroids(search_range=tracking_range, memory=int(memory))
        except Exception as e:
            print(e)
            self.statusBar().showMessage(f'Error tracking centroids: {e}', 4000)
            return

        print(f'Tracked centroids for stack {self.stack.name}')
        self.left_toolbar.tracking_range.setText(f'{self.stack.tracking_range:.2f}')
        self.statusBar().showMessage(f'Tracked centroids for stack {self.stack.name}.', 2000)
        self._update_tracking_overlay()
        self._recolor_tracks()
        self.canvas.draw_masks()
        self.left_toolbar.propagate_FUCCI_checkbox.setEnabled(True)
        self.left_toolbar.also_save_tracking.setChecked(True)

    def track_centroids(self, **kwargs):
        """
        Track centroids for the current stack using trackpy.

        Parameters
        ----------
        search_range : float
            The maximum distance a cell can move between frames. If None, the search range is determined automatically.
        v_quantile : float
            The quantile to use for automatically determining the search range. Defaults to 0.97.
            Adjust this lower if too many cells are being linked incorrectly, or higher if you're losing tracks.
        memory : int
            The maximum number of frames a cell can be lost for before being considered a new cell. Defaults to 0.
        """

        needs_outlines = []
        for frame in self.stack.frames:
            if not frame.has_outlines:
                needs_outlines.append(frame)

        for frame in self._progress_bar(needs_outlines, desc='Generating outlines'):
            outlines = utils.outlines_list(frame.masks)
            for cell, outline in zip(frame.cells, outlines):
                cell.outline = outline
                cell.get_centroid()
            frame.has_outlines = True

        self.stack.track_centroids(**kwargs)

    def _recolor_tracks(self):
        # recolor cells so each particle has one color over time
        for frame in self.stack.frames:
            if hasattr(frame, 'stored_mask_overlay'):
                del frame.stored_mask_overlay

        t = self.stack.tracked_centroids
        colors = self.canvas.random_color_ID(t['particle'].max() + 1)
        t['color'] = colors[t['particle']]
        for frame in self.stack.frames:
            tracked_frame = t[t.frame == frame.frame_number].sort_values('cell_number')
            frame.set_cell_attrs('color_ID', self.canvas.cell_cmap(tracked_frame['color']))

        self.canvas.draw_masks_parallel()

    def particle_from_cell(self, cell_number, frame_number=None):
        """
        Get the particle number for the specified cell in the specified frame.

        Parameters
        ----------
        cell_number : int
            The cell number to get the particle number for.
        frame_number : int
            The frame number to get the particle number for. If None, the current frame is used.
        """
        if not hasattr(self.stack, 'tracked_centroids'):
            return None
        if frame_number is None:
            frame_number = self.frame_number
        if cell_number is None:
            return None

        return self.stack.particle_from_cell(cell_number, frame_number)

    def cell_from_particle(self, particle, frame_number=None):
        """
        Get the cell number for the specified particle in the specified frame.

        Parameters
        ----------
        particle : int
            The particle number to get the cell number for.
        frame_number : int
            The frame number to get the cell number for. If None, the current frame is used.
        """
        if not hasattr(self.stack, 'tracked_centroids'):
            return None
        if frame_number is None:
            frame_number = self.frame_number
        if particle is None:
            return None

        return self.stack.cell_from_particle(particle, frame_number)

    def split_particle_tracks(self):
        """
        Split the selected particle track at the current frame. The previous timepoints will keep the original particle number.

        Returns
        -------
        new_particle : int
            The new particle number assigned to the second half of the split particle.
            If no split was made, returns None.
        """
        new_particle = self.stack.split_particle_track(self.selected_particle_n, self.frame_number)

        if new_particle is None:
            return None

        self.selected_particle_n = new_particle

        # assign a random color to the new particle
        new_color = self.canvas.random_cell_color()
        for cell in self.stack.get_particle(self.selected_particle_n):
            cell.color_ID = new_color
            frame = self.stack.frames[cell.frame]
            if hasattr(frame, 'stored_mask_overlay'):
                self.canvas.add_cell_highlight(cell.n, frame=frame, color=new_color, alpha=self.canvas.masks_alpha, layer='mask')

        print(f'Split particle {self.selected_particle_n} at frame {self.frame_number}')

        self.left_toolbar.also_save_tracking.setChecked(True)
        self._plot_particle_statistic()
        self._update_tracking_overlay()

        return new_particle

    def merge_particle_tracks(self, first_particle, second_particle):
        """
        Merge the two specified particles at the current frame.
        This will combine two particle IDs and create up to two new ones for the severed ends of the old particle tracks.

        Parameters
        ----------
        first_particle : int
            The first particle to merge.
        second_particle : int
            The second particle to merge.
        """

        if hasattr(self.stack, 'tracked_centroids'):
            if first_particle == second_particle:  # same particle, no need to merge
                return
            else:
                first_particle_cell = self.stack.get_particle(first_particle)[0]
                if (
                    first_particle_cell.frame == self.frame_number
                ):  # merging parent only appears in current frame. Merging not possible.
                    return
                merged_color = first_particle_cell.color_ID
                merged, new_head, new_tail = self.stack.merge_particle_tracks(first_particle, second_particle, self.frame_number)
                self.left_toolbar.also_save_tracking.setChecked(True)
                if new_head is not None:
                    new_head_color = self.canvas.random_cell_color()
                    for cell in self.stack.get_particle(new_head):
                        cell.color_ID = new_head_color
                        if hasattr(self.stack.frames[cell.frame], 'stored_mask_overlay'):
                            self.canvas.add_cell_highlight(
                                cell.n,
                                frame=self.stack.frames[cell.frame],
                                color=new_head_color,
                                alpha=self.canvas.masks_alpha,
                                layer='mask',
                            )
                if new_tail is not None:
                    new_tail_color = self.canvas.random_cell_color()
                    for cell in self.stack.get_particle(new_tail):
                        cell.color_ID = new_tail_color
                        if hasattr(self.stack.frames[cell.frame], 'stored_mask_overlay'):
                            self.canvas.add_cell_highlight(
                                cell.n,
                                frame=self.stack.frames[cell.frame],
                                color=new_tail_color,
                                alpha=self.canvas.masks_alpha,
                                layer='mask',
                            )

                for cell in self.stack.get_particle(merged):
                    cell.color_ID = merged_color
                    if hasattr(self.stack.frames[cell.frame], 'stored_mask_overlay'):
                        self.canvas.add_cell_highlight(
                            cell.n,
                            frame=self.stack.frames[cell.frame],
                            color=merged_color,
                            alpha=self.canvas.masks_alpha,
                            layer='mask',
                        )

                print(f'Merged particles {first_particle} and {second_particle} at frame {self.frame_number}')
                self._plot_particle_statistic()
                self._update_tracking_overlay()
                current_cell = self.cell_from_particle(merged)
                self.canvas.add_cell_highlight(current_cell, color=merged_color, alpha=self.canvas.masks_alpha, layer='mask')

                new_tail_cell = self.cell_from_particle(new_tail)
                if new_tail_cell is not None:
                    self.canvas.add_cell_highlight(
                        new_tail_cell, color=new_tail_color, alpha=self.canvas.masks_alpha, layer='mask'
                    )

    def _set_LUTs(self, refresh=True):
        """Set the LUTs for the image display based on the current slider values."""
        self.canvas.img.setLevels(self.left_toolbar.LUT_slider_values, refresh=refresh)
        self.left_toolbar.update_LUT_labels()

    def _update_display(self):
        """Redraw the image data with whatever new settings have been applied from the toolbar."""
        if not self.file_loaded:
            return
        self._show_seg_overlay()
        img_data = self.frame.img
        seg_data = self.canvas.image_transform(self.frame.outlines)
        self._normalize()
        self.canvas.update_display(img_data=img_data, seg_data=seg_data, RGB_checks=self.left_toolbar.RGB_visible)

    def _autorange_LUT_sliders(self):
        if self.is_grayscale:
            n_colors = 1
        else:
            n_colors = 3

        if self.is_zstack:
            all_imgs = np.array([frame.zstack for frame in self.stack.frames]).reshape(-1, n_colors)
        else:
            all_imgs = np.array([frame.img for frame in self.stack.frames]).reshape(-1, n_colors)

        if len(all_imgs) > 1e6:
            # downsample to speed up calculation
            all_imgs = all_imgs[:: len(all_imgs) // int(1e6)]
        stack_range = np.array([np.min(all_imgs, axis=0), np.max(all_imgs, axis=0)]).T
        self.stack.min_max = stack_range
        self.left_toolbar.set_LUT_slider_ranges(stack_range)

    def _update_voxel_size(self):
        if not self.file_loaded:
            return
        xy = self.left_toolbar.xy_size
        z = self.left_toolbar.z_size

        if xy is not None:
            for frame in self.stack.frames:
                frame.scale = xy
        if z is not None:
            for frame in self.stack.frames:
                frame.z_scale = z

    def _update_voxel_size_labels(self):
        """Update the labels next to the voxel size boxes with the current values."""
        if hasattr(self.frame, 'scale'):
            xy_size = self.frame.scale
            self.left_toolbar.xy_size = xy_size
        if hasattr(self.frame, 'z_scale'):
            z_size = self.frame.z_scale
            self.left_toolbar.z_size = z_size

    def _normalize(self):
        execution_times = {}

        start_time = time.time()
        execution_times['channel check'] = time.time() - start_time

        start_time = time.time()
        if self.left_toolbar.normalize_type == 'frame':  # normalize the frame
            if not hasattr(self.frame, 'bounds'):
                self.frame.bounds = self._get_frame_bounds(self.frame)

            if self.is_zstack:
                bounds = self.frame.bounds[self.zstack_number]
            else:
                bounds = self.frame.bounds

        elif self.left_toolbar.normalize_type == 'stack':  # normalize the stack
            if not hasattr(self.stack, 'bounds'):
                self.stack.bounds = self._get_stack_bounds()

            bounds = self.stack.bounds

        else:  # custom: use the slider values
            bounds = np.array([slider.value() for slider in self.left_toolbar.LUT_range_sliders])

        execution_times['bounds retrieval'] = time.time() - start_time

        start_time = time.time()
        self.left_toolbar.LUT_slider_values = bounds
        execution_times['self.left_toolbar.LUT_slider_values = bounds'] = time.time() - start_time

        # Print all execution times sorted by duration
        if debug_execution_times:
            print('-------------NORMALIZE-------------')
            sorted_execution_times = sorted(execution_times.items(), key=lambda item: item[1], reverse=True)
            for description, duration in sorted_execution_times:
                print(f'{description}: {duration:.4f} seconds')

        return bounds

    def _get_frame_bounds(self, frame):
        """Get the bounds of the frame for normalization."""
        if hasattr(frame, 'zstack'):
            img = frame.zstack
            is_grayscale = img.ndim == 3
        else:
            img = frame.img.reshape(1, *frame.img.shape)
            is_grayscale = img.ndim == 2

        if is_grayscale:
            img = img.reshape(*img.shape, 1)

        bounds=[]
        for z_slice in img:
            bounds.append(get_quantile(z_slice, q=(1, 99), mask_zeros=True))
        return np.squeeze(bounds)

    def _precompute_bounds(self, frames=None):
        if not self.file_loaded:
            return
        
        if frames is None:
            frames = self.stack.frames
        
        if N_CORES == 1:
            return
        else:
            self.bounds_processor.process_frames(frames)
        
    def _get_stack_bounds(self):
        """Get the bounds of the stack for normalization."""
        if not self.file_loaded:
            return
        first_frame = self.stack.frames[0]
        if hasattr(first_frame, 'zstack'):
            shape = first_frame.zstack.shape
            is_grayscale = len(shape) == 3
        else:
            shape = first_frame.img.shape
            is_grayscale = len(shape) == 2

        img_size = np.prod(shape)
        downsample_factor = img_size // int(1e6)

        if downsample_factor == 0:
            downsample_factor = 1

        all_imgs = []
        for frame in self.stack.frames:
            if hasattr(frame, 'zstack'):
                img = frame.zstack
                img = img[:, ::downsample_factor]
            else:
                img = frame.img
                img = img[::downsample_factor]
            if is_grayscale:
                img = img.reshape(*img.shape, 1)
            all_imgs.append(img)
        all_imgs = np.stack(all_imgs)
        bounds = get_quantile(all_imgs, q=(1, 99), mask_zeros=True)

        return bounds

    def _open_command_line(self):
        # Create a separate window for the command line interface
        if not hasattr(self, 'cli_window') or not self.cli_window.isVisible():
            self.cli_window = CommandLineWindow(self, self.globals_dict, self.locals_dict)
            self.globals_dict['cli'] = self.cli_window.cli
            self.cli_window.show()
        else:
            self.cli_window.activateWindow()
            self.cli_window.raise_()

    def _open_script_editor(self):
        # Create a separate window for the script editor
        if not hasattr(self, 'script_window') or not self.script_window.isVisible():
            self.script_window = ScriptWindow(self, self.globals_dict, self.locals_dict)
            self.script_window.show()

    def select_cell(self, particle=None, cell=None):
        """
        Select a cell or particle by number. If None, selection is cleared.

        Parameters
        ----------
        particle : int
            The particle number to select.
        cell : int
            The cell number to select. If both particle and cell are specified, cell takes precedence.
        """

        if self.FUCCI_mode:  # classifying FUCCI, no cell selection
            return

        if cell is not None:  # select by cell number
            self.selected_cell_n = cell
            self.selected_particle_n = self.particle_from_cell(cell)
        elif particle is not None:  # select by particle number
            self.selected_particle_n = particle
            self.selected_cell_n = self.cell_from_particle(particle)
        else:  # clear selection
            self.selected_cell_n = None
            self.selected_particle_n = None

        # update labels
        self._update_cell_label(self.selected_cell_n)
        self._update_tracking_ID_label(self.selected_particle_n)
        self._plot_particle_statistic()  # put info about the particle in the right toolbar

        self.canvas.clear_overlay('selection')

        if self.selected_cell_n is None:
            self.cell_properties_label.setText('')  # clear the cell attributes table
            return

        # highlight cell
        self.canvas.add_cell_highlight(
            self.selected_cell_n, alpha=self.canvas.selected_cell_alpha, color=self.canvas.selected_cell_color
        )

        # show cell attributes in right toolbar
        if len(self.selected_cell.outline) > 0:
            labels = sorted(self._cell_stat_attrs(self.selected_cell))
            attrs = [getattr(self.selected_cell, attr) for attr in labels]
            cell_attrs_label = create_html_table(labels, attrs)
        else:
            cell_attrs_label = ''
        self.cell_properties_label.setText(cell_attrs_label)

    def _clear_particle_statistic(self):
        self.particle_stat_plot.clear()
        self.particle_stat_plot.setLabel('left', '')
        self.particle_stat_plot.addItem(self.stat_plot_frame_marker)

    def _new_histogram(self):
        self._plot_histogram()
        self.histogram.autoRange()

    def _plot_histogram(self):
        if not self.file_loaded:
            return
        if not self.stat_tabs.currentIndex() == 0:
            return
        self.histogram.clear()
        hist_attr = self.histogram_menu.currentText()
        self.histogram.setLabel('bottom', hist_attr)
        if hist_attr == 'Select Cell Attribute':
            return
        # get the attribute values
        # TODO: check whether to operate on stack or frame
        cell_attrs = np.array(self.frame.get_cell_attrs(hist_attr, fill_value=np.nan))

        if np.all(np.isnan(cell_attrs)):
            return

        hist_data = np.array(cell_attrs)[~np.isnan(cell_attrs)]

        iqr = np.percentile(hist_data, 75) - np.percentile(hist_data, 25)
        bin_width = 2 * iqr / (len(hist_data) ** (1 / 3))
        bins = np.arange(np.min(hist_data), np.max(hist_data) + bin_width, bin_width)

        n, bins = np.histogram(hist_data, bins=bins, density=True)
        self.histogram.plot(bins, n, stepMode=True, fillLevel=0, brush=(0, 0, 255, 150))

    def _plot_particle_statistic(self):
        if not self.file_loaded or not hasattr(self.stack, 'tracked_centroids'):
            return
        if not self.stat_tabs.currentIndex() == 1:
            return
        measurement = self.particle_stat_menu.currentText()

        self.particle_stat_plot.clear()  # clear the plot
        self.particle_stat_plot.setLabel('left', measurement)
        if measurement == 'Select Cell Attribute':
            return

        self.particle_stat_plot.addItem(self.stat_plot_frame_marker)  # add the frame marker line
        if self.selected_particle_n is not None:
            color = pg.mkColor(np.array(self.canvas.cell_cmap(0))[:3] * 255)
            timepoints = self.stack.get_particle_attr(self.selected_particle_n, 'frame')
            if measurement == 'cell_cycle':  # fetch up-to-date cell cycle classification
                green, red = np.array(
                    self.stack.get_particle_attr(self.selected_particle_n, ['green', 'red'], fill_value=False)
                ).T
                values = green + 2 * red
            else:
                values = self.stack.get_particle_attr(self.selected_particle_n, measurement, fill_value=np.nan)
            if np.all(np.isnan(values)):  # no data to plot
                return
            self.particle_stat_plot.plot(
                timepoints, values, pen=color, symbol='o', symbolPen='w', symbolBrush=color, symbolSize=7, width=4
            )
            self.particle_stat_plot.autoRange()

    def _plot_time_series(self):
        if not self.file_loaded:
            return
        if not self.stat_tabs.currentIndex() == 2:
            return
        measurement = self.time_series_menu.currentText()
        self.time_series_plot.clear()
        self.time_series_plot.setLabel('left', measurement)
        self.time_series_plot.addItem(self.time_series_frame_marker)

        if measurement == 'Select Cell Attribute':
            return
        # get the attribute values
        quantiles = []
        frames = np.arange(len(self.stack.frames))
        for frame in self.stack.frames:
            frame_attrs = np.array(frame.get_cell_attrs(measurement, fill_value=np.nan))
            frame_quantiles = np.nanquantile(frame_attrs, (0.25, 0.5, 0.75))
            quantiles.append(frame_quantiles)
        quantiles = np.array(quantiles)

        if np.all(np.isnan(quantiles)):
            return

        median = quantiles[:, 1]
        bottom = median - quantiles[:, 0]
        top = quantiles[:, 2] - median

        median_pen = pg.mkPen((255, 255, 255, 255))
        quantile_pen = pg.mkPen((100, 100, 255, 255))
        error_bars = pg.ErrorBarItem(x=frames, y=median, top=top, bottom=bottom, pen=quantile_pen, beam=0.5)
        self.time_series_plot.addItem(error_bars)
        self.time_series_plot.plot(frames, median, pen=median_pen)
        self.time_series_plot.autoRange()

    def _FUCCI_click(self, event, current_cell_n):
        if current_cell_n >= 0:
            cell = self.frame.cells[current_cell_n]
            if event.button() == Qt.MouseButton.LeftButton:
                self._classify_cell_cycle(cell, 0)
            if event.button() == Qt.MouseButton.RightButton:
                self._classify_cell_cycle(cell, 1)
            if event.button() == Qt.MouseButton.MiddleButton:
                self._classify_cell_cycle(cell, 2)
        else:
            self.select_cell(None)

    def _start_cell_split(self, event):
        self.drawing_cell_split = True
        self.cell_split.clearPoints()

        x, y = self.canvas.get_plot_coords(event.scenePos(), pixels=True)
        # Add the first handle
        self.cell_split.add_vertex(y, x)

    def _start_drawing_segmentation(self, event):
        self.drawing_cell_roi = True
        self.cell_roi.clearPoints()

        x, y = self.canvas.get_plot_coords(event.scenePos(), pixels=True)
        # Add the first handle
        self.cell_roi.add_vertex(y, x)
        self.cell_roi.first_handle_pos = np.array((y, x))
        self.cell_roi.last_handle_pos = np.array((y, x))

        self.roi_is_closeable = False

    def _on_click(self, event):
        if not self.file_loaded:
            return

        x, y = self.canvas.get_plot_coords(event.scenePos(), pixels=True)
        current_cell_n = self._get_cell(x, y)
        if hasattr(self.stack, 'tracked_centroids'):
            current_particle_n = self.particle_from_cell(current_cell_n)
            self.t2 = self.stack.tracked_centroids.copy()  ### to delete
        else:
            current_particle_n = None

        if self.FUCCI_mode:  # cell cycle classification
            self._FUCCI_click(event, current_cell_n)

        else:
            if event.button() == Qt.MouseButton.RightButton:
                if self.drawing_cell_roi:
                    self._close_cell_roi()
                elif self.drawing_cell_split:
                    self._split_cell()

                elif event.modifiers() == Qt.KeyboardModifier.ShiftModifier:  # split selected particle at current frame
                    self.selected_particle_n = self.particle_from_cell(current_cell_n)
                    if self.selected_particle_n is not None:
                        self.split_particle_tracks()
                elif event.modifiers() == Qt.KeyboardModifier.AltModifier:  # start drawing cell split line
                    self._start_cell_split(event)
                else:  # start drawing new segmentation ROI
                    self._start_drawing_segmentation(event)

            elif event.button() == Qt.MouseButton.LeftButton:
                # cancel right-click actions
                if self.drawing_cell_roi:
                    self.cell_roi.clearPoints()
                    self.drawing_cell_roi = False
                elif self.drawing_cell_split:
                    self.cell_split.clearPoints()
                    self.drawing_cell_split = False
                elif self.mitosis_mode:
                    self._mitosis_selected(self.particle_from_cell(current_cell_n))

                # cell selection actions
                if current_cell_n >= 0:
                    if (event.modifiers() & Qt.KeyboardModifier.ControlModifier) and (
                        event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                    ):
                        # delete cell from all frames
                        particle = self.particle_from_cell(current_cell_n, self.frame_number)
                        if particle is not None:
                            self.delete_particle(particle_n=particle)
                        self.select_cell(None)
                        self._update_display()

                    elif event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                        # delete cell from current frame
                        self.delete_cell_mask(current_cell_n)
                        if current_particle_n is None:
                            self.select_cell(None)  # deselect the cell
                        else:
                            self.select_cell(particle=current_particle_n)

                    elif (event.modifiers() & Qt.KeyboardModifier.AltModifier) and (
                        event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                    ):
                        # merge cells in all frames
                        particle = self.particle_from_cell(current_cell_n, self.frame_number)
                        if particle is not None:
                            self.merge_particle_masks(self.selected_particle_n, particle)
                        self.select_cell(particle=self.selected_particle_n)  # reselect the merged particle

                    elif event.modifiers() == Qt.KeyboardModifier.AltModifier and self.selected_cell_n is not None:
                        self.merge_cell_masks(self.selected_cell_n, current_cell_n)
                        self.select_cell(cell=self.selected_cell_n)  # reselect the merged cell

                    elif event.modifiers() == Qt.KeyboardModifier.ShiftModifier:  # merge particles
                        if self.selected_particle_n is not None:
                            second_particle = self.particle_from_cell(current_cell_n)
                            if second_particle is not None:  # if a particle is found
                                self.merge_particle_tracks(self.selected_particle_n, second_particle)
                        self.select_cell(cell=current_cell_n)

                    elif current_cell_n == self.selected_cell_n:
                        # clicking the same cell again deselects it
                        self.select_cell(None)

                    else:
                        # select the cell
                        self.select_cell(cell=current_cell_n)

                else:  # clicked on background, deselect
                    self.select_cell(None)
        self._scan_tracked_centroids()  ### to delete

    def _classify_cell_cycle(self, cell, classification):
        if classification == 0:
            cell.green = not cell.green
        elif classification == 1:
            cell.red = not cell.red
        else:
            if cell.green and cell.red:
                cell.green = False
                cell.red = False
            else:
                cell.green = True
                cell.red = True

        if self.left_toolbar.propagate_FUCCI_checkbox.isChecked():
            if hasattr(self.stack, 'tracked_centroids'):
                particle = self.stack.get_particle(cell)
                for cell_timepoint in particle:
                    if cell_timepoint.frame > cell.frame:
                        cell_timepoint.green = cell.green
                        cell_timepoint.red = cell.red

        if self.FUCCI_mode:
            overlay_color = self.FUCCI_dropdown
            if overlay_color == 3:
                color = ['none', 'g', 'r', 'orange'][2 * cell.red + cell.green]
            elif overlay_color == 1:
                color = ['none', 'g'][cell.green]
            elif overlay_color == 2:
                color = ['none', 'r'][cell.red]
            else:
                color = 'none'
            self.canvas.add_cell_highlight(cell.n, alpha=1, color=color, img_type='outlines', layer='FUCCI')

        self._plot_particle_statistic()

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

    def _mouse_moved(self, pos):
        """Dynamically update the cell mask overlay as the user draws a new cell."""
        if not self.file_loaded:
            return

        if self.drawing_cell_split:
            x, y = self.canvas.get_plot_coords(pos, pixels=True)  # position in plot coordinates

            self.cell_split.add_vertex(y, x)

        elif self.drawing_cell_roi:
            x, y = self.canvas.get_plot_coords(pos, pixels=True)  # position in plot coordinates

            if np.array_equal((y, x), self.cell_roi.last_handle_pos):
                return
            else:
                self.cell_roi.add_vertex(y, x)
                if self.roi_is_closeable:
                    if np.linalg.norm(np.array((y, x)) - self.cell_roi.first_handle_pos) < 3:
                        self._close_cell_roi()
                        return
                else:
                    if np.linalg.norm(np.array((y, x)) - self.cell_roi.first_handle_pos) > 3:
                        self.roi_is_closeable = True

    def _get_cell(self, x, y):
        """Get the cell number at a given pixel coordinate."""
        if x < 0 or y < 0 or x >= self.canvas.img_data.shape[0] or y >= self.canvas.img_data.shape[1]:
            return -1  # out of bounds
        cell_n = self.frame.masks[x, y]
        if cell_n == 0:
            return -1
        else:
            return cell_n - 1

    def _split_cell(self):
        self.drawing_cell_split = False
        self.split_cell_masks()
        self.cell_split.clearPoints()
        self.select_cell(particle=self.selected_particle_n)
        self._update_display()

    def _close_cell_roi(self):
        """Close the cell ROI and add the new cell mask to the frame."""
        self.drawing_cell_roi = False
        enclosed_pixels = self.cell_roi.get_enclosed_pixels()
        # remove pixels outside the image bounds
        enclosed_pixels = enclosed_pixels[
            (enclosed_pixels[:, 0] >= 0)
            & (enclosed_pixels[:, 0] < self.frame.masks.shape[0])
            & (enclosed_pixels[:, 1] >= 0)
            & (enclosed_pixels[:, 1] < self.frame.masks.shape[1])
        ]
        new_mask_n = self.add_cell_mask(enclosed_pixels)
        if new_mask_n is not False:
            print(f'Added cell {new_mask_n}')
        self.cell_roi.clearPoints()
        self._update_display()

    def random_color(self, n_samples=None):
        """
        Generate a collection of random cell mask colors.

        Parameters
        ----------
        n_samples : int
            The number of colors to generate. If None, a single color is generated.
        """

        random_colors = np.random.randint(0, self.canvas.cell_n_colors, size=n_samples)
        colors = np.array(self.canvas.cell_cmap(random_colors))[..., :3]

        return colors

    @property
    def is_grayscale(self):
        return self.left_toolbar.is_grayscale

    @is_grayscale.setter
    def is_grayscale(self, value):
        if value:
            self.left_toolbar.grayscale_mode()
        else:
            self.left_toolbar.RGB_mode()

    # --------------Segmentation functions----------------
    def add_cell_mask(self, enclosed_pixels):
        """
        Add a new cell mask to the current frame based on the enclosed pixels.

        Parameters
        ----------
        enclosed_pixels : np.ndarray
            An array of pixel coordinates enclosed by the new cell mask.

        Returns
        -------
        new_mask_n : int
            The cell number of the new mask.
        """

        new_mask_n = self.frame.n_cells  # new cell number
        cell_mask = np.zeros_like(self.frame.masks, dtype=bool)
        cell_mask[enclosed_pixels[:, 0], enclosed_pixels[:, 1]] = True
        new_mask = cell_mask & (self.frame.masks == 0)

        if new_mask.sum() < 5:  # check if the mask is more than 4 pixels (minimum for cellpose to generate an outline)
            return False

        self.frame.masks[new_mask] = new_mask_n + 1

        cell_color_n = self.canvas.random_color_ID()
        cell_color = self.canvas.cell_cmap(cell_color_n)
        if self.frame.has_outlines:
            outline = utils.outlines_list(new_mask)[0]
        else:
            outline = np.empty((0, 2), dtype=int)

        self.frame.outlines[outline[:, 1], outline[:, 0]] = True
        centroid = np.mean(enclosed_pixels, axis=0)
        self.add_cell(
            new_mask_n, outline, color_ID=cell_color, centroid=centroid, frame_number=self.frame_number, red=False, green=False
        )

        if hasattr(self.stack, 'tracked_centroids'):
            t = self.stack.tracked_centroids
            new_particle_ID = t['particle'].max() + 1
            new_particle = pd.DataFrame(
                [[new_mask_n, centroid[0], centroid[1], self.frame_number, new_particle_ID, cell_color_n]],
                columns=t.columns,
                index=[t.index.max() + 1],
            )  # TODO: handle situations where tracked_centroids has more or fewer columns
            self.stack.tracked_centroids = pd.concat([t, new_particle])
            self.stack.tracked_centroids = self.stack.tracked_centroids.sort_values(['frame', 'particle'])
            self.left_toolbar.also_save_tracking.setChecked(True)

        self.canvas.draw_outlines()
        self._update_tracking_overlay()
        self.canvas.add_cell_highlight(new_mask_n, alpha=self.canvas.masks_alpha, color=cell_color, layer='mask')

        self._update_ROIs_label()
        return new_mask_n

    def split_cell_masks(self, min_size=5):
        """
        Split cells that intersect with the current curve.

        Parameters
        ----------
        min_size : int
            The minimum size of a cell mask in pixels. Cells smaller than this will be merged with their largest neighbor.
        """

        curve_coords = np.array([(p.x(), p.y()) for p in self.cell_split.points]).astype(int)
        next_label = np.max(self.frame.masks) + 1

        # Create a binary mask of the curve
        curve_mask = np.zeros_like(self.frame.masks, dtype=bool)
        for i in range(len(curve_coords) - 1):
            rr, cc = draw.line(curve_coords[i][0], curve_coords[i][1], curve_coords[i + 1][0], curve_coords[i + 1][1])
            # remove out-of-bounds coordinates
            inbound_coords = (cc >= 0) & (cc < curve_mask.shape[0]) & (rr >= 0) & (rr < curve_mask.shape[1])
            rr, cc = rr[inbound_coords], cc[inbound_coords]
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

            neighbor_labels = []
            dilated = component_mask
            counter = 0
            while len(neighbor_labels) == 0:
                neighbor_labels, dilated = find_neighbors(dilated, labels_array)
                counter += 1
                if counter > max_iter:
                    raise ValueError('No neighbors found')

            # Count occurrences of each neighbor label
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            return unique_labels[np.argmax(counts)]

        split = False
        new_masks = self.frame.masks.copy()
        for orig_label in intersected_labels:
            # Get the current region
            region_mask = self.frame.masks == orig_label

            # Create temporary binary mask with the curve
            temp_mask = region_mask.copy()
            curve_pixels = curve_mask & region_mask
            temp_mask[curve_pixels] = False

            # Label connected components
            labeled_parts, num_features = ndimage.label(temp_mask)

            if num_features < 2:  # Didn't split the label, move on to next candidate
                continue

            # Get sizes of all components
            component_sizes = np.array([np.sum(labeled_parts == i) for i in range(1, num_features + 1)])

            num_labels = component_sizes >= min_size

            if np.sum(num_labels) < 2:  # only one component is above the minimum size
                continue

            split = True
            # Find largest component to keep original label
            largest_idx = np.argmax(component_sizes) + 1

            # Clear original region
            new_masks[region_mask] = 0
            self.frame.masks[region_mask] = 0  # clear the original mask, to be replaced after split computation

            # largest component gets original label
            new_masks[labeled_parts == largest_idx] = orig_label

            # Process other components
            other_indices = [i + 1 for i in range(num_features) if i + 1 != largest_idx]

            new_labels = []
            for comp_idx in other_indices:  # assign new labels to components above minimum size
                if component_sizes[comp_idx - 1] >= min_size:
                    new_labels.append(next_label)
                    new_masks[labeled_parts == comp_idx] = next_label
                    next_label += 1
                else:  # merge small components with their largest neighbors
                    component = labeled_parts == comp_idx
                    try:
                        neighbor_label = find_largest_neighbor_label(component, self.frame.masks)
                    except ValueError:
                        neighbor_label = None

                    if neighbor_label is not None:
                        new_masks[component] = neighbor_label

            print(f'Split cell {orig_label - 1}, new labels: {", ".join(str(n - 1) for n in new_labels)}')

            # Handle curve pixels by assigning them to the most connected component
            curve_points = np.where(curve_pixels)
            for y, x in zip(*curve_points):
                neighborhood_slice = (
                    slice(max(0, y - 1), min(self.frame.masks.shape[0], y + 2)),
                    slice(max(0, x - 1), min(self.frame.masks.shape[1], x + 2)),
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

            old_cell = self.frame.cells[orig_label - 1]
            old_mask = new_masks == orig_label
            if hasattr(old_cell, '_centroid'):
                del old_cell._centroid
            outline = self.frame.add_outline(old_mask)
            self.frame.masks[old_mask] = orig_label  # restore the original label to the largest mask
            if self.frame.has_outlines:
                old_cell.outline = outline

            for new_label in new_labels:
                enclosed_pixels = np.array(np.where(new_masks == new_label)).T
                if len(enclosed_pixels) > 0:
                    self.add_cell_mask(enclosed_pixels)

        if split:
            del self.frame.stored_mask_overlay

    def add_cell(self, n, outline, color_ID=None, red=False, green=False, frame_number=None, **kwargs):
        """
        Add a new cell to the current frame.

        Parameters
        ----------
        n : int
            The cell number.
        outline : np.ndarray
            The outline of the cell mask.
        color_ID : int
            The color ID of the cell mask. If None, a random color is generated.
        frame_number : int
            The frame number to add the cell to. If None, the current frame is used.

        Any other keyword arguments are passed to the Cell constructor.

        Returns
        -------
        new_cell : Cell
            The new cell object.
        """
        if frame_number is None:
            frame_number = self.frame_number
        else:
            frame = self.stack.frames[frame_number]

        new_cell = Cell(
            n,
            outline,
            color_ID=color_ID,
            red=red,
            green=green,
            frame_number=frame_number,
            parent=self.stack.frames[self.frame_number],
            **kwargs,
        )
        frame.cells = np.append(self.frame.cells, new_cell)
        frame.n_cells += 1

        return new_cell

    def merge_cell_masks(self, cell_n1, cell_n2, frame_number=None):
        """
        Merges the cell_n2 mask into cell_n1.
        This is done by reassigning the cell_n2 mask to cell_n1 and deleting cell_n2.

        Parameters
        ----------
        cell_n1 : int
            The cell number to merge into.
        cell_n2 : int
            The cell number to merge.
        frame_number : int
            The frame number to merge the cells in. If None, the current frame is used.
        """

        if cell_n1 == cell_n2:
            return

        if cell_n1 > cell_n2:
            selected_cell_n = cell_n1 - 1  # new cells will be renumbered
        else:
            selected_cell_n = cell_n1

        if frame_number is None:
            frame_number = self.frame_number
        # purge cell 2
        new_cell, _ = self.stack.merge_cells(cell_n1, cell_n2, frame_number=frame_number)
        print(f'Merged cell {cell_n2} into cell {cell_n1} in frame {frame_number}')

        if hasattr(self.stack, 'tracked_centroids'):
            self.left_toolbar.also_save_tracking.setChecked(True)

        # add new cell mask to the overlay
        self.canvas.add_cell_highlight(
            selected_cell_n,
            alpha=self.canvas.masks_alpha,
            color=new_cell.color_ID,
            layer='mask',
            frame=self.stack.frames[frame_number],
        )

        if frame_number == self.frame_number:
            self._update_tracking_overlay()
            self.select_cell(cell=selected_cell_n)
            self._update_ROIs_label()
            self._update_display()

        # self.check_cell_numbers() # for troubleshooting

    def merge_particle_masks(self, particle_n1, particle_n2):
        """
        Merges the masks of particle_n2 into particle_n1 in every frame.
        This is done by a series of cell mask merges.

        Parameters
        ----------
        particle_n1 : int
            The particle number to merge into.
        particle_n2 : int
            The particle number to merge.
        """
        t = self.stack.tracked_centroids
        # renumber tracked_centroids so particle_n2 is the largest number
        remapped_particles = np.arange(t['particle'].max() + 2)
        remapped_particles = np.delete(remapped_particles, particle_n2)
        t['particle'] = remapped_particles[t['particle']]
        particle_n1 = remapped_particles[particle_n1]
        particle_n2 = remapped_particles[particle_n2]

        # get cells for each particle
        particle1 = self.stack.get_particle(particle_n1)
        particle2 = self.stack.get_particle(particle_n2)

        particle1_frames = [cell.frame for cell in particle1]
        particle2_frames = [cell.frame for cell in particle2]

        particle1_color = particle1[0].color_ID

        merge_frames = set(particle1_frames).intersection(
            particle2_frames
        )  # frames where particle2 masks need to be merged into particle1 masks
        relabel_frames = set(particle2_frames) - merge_frames  # frames where particle2 cell needs to be relabeled as particle1

        for frame_number in merge_frames:
            cell1 = particle1[particle1_frames.index(frame_number)]
            cell2 = particle2[particle2_frames.index(frame_number)]
            self.merge_cell_masks(cell1.n, cell2.n, frame_number=frame_number)

        for frame_number in relabel_frames:
            frame = self.stack.frames[frame_number]
            # relabel particle, redraw color
            cell2 = particle2[particle2_frames.index(frame_number)]
            t.loc[t['particle'] == particle_n2, 'particle'] = particle_n1
            cell2.color_ID = particle1_color
            if hasattr(frame, 'stored_mask_overlay'):
                self.canvas.add_cell_highlight(
                    cell2.n, alpha=self.canvas.masks_alpha, color=particle1_color, layer='mask', frame=frame
                )
        if len(relabel_frames) > 0:
            print(f'Relabeled cell {cell2.n} as particle {particle_n1} in frames {relabel_frames}')

    def _check_cell_numbers(self):
        """for troubleshooting: check if the cell numbers in the frame and the masks align."""
        cell_number_alignment = np.array([cell.n != n for n, cell in enumerate(self.frame.cells)])
        if np.any(cell_number_alignment):
            print(f'{np.sum(cell_number_alignment)} cell numbers misalign starting with {np.where(cell_number_alignment)[0][0]}')

        mask_number_alignment = np.array([n != mask_n for n, mask_n in enumerate(fastremap.unique(self.frame.masks))])
        if np.any(mask_number_alignment):
            print(f'{np.sum(mask_number_alignment)} cell masks misalign starting with {np.where(mask_number_alignment)[0][0]}')

    def _generate_outlines(self):
        if not self.file_loaded:
            return

        if self.left_toolbar.segment_on_stack.isChecked():
            frames = self.stack.frames
        else:
            frames = [self.frame]

        self.generate_outlines(frames)
        self.statusBar().showMessage('Generated outlines.', 1000)

    def generate_outlines(self, frames):
        """
        Generate cell outlines for a list of frames.

        Parameters
        ----------
        frames : list
            A list of Frame objects to generate outlines for.
        """

        for frame in self._progress_bar(frames, desc='Generating outlines'):
            outlines = utils.outlines_list(frame.masks)
            for cell, outline in zip(frame.cells, outlines):
                cell.outline = outline
                cell.get_centroid()
            frame.has_outlines = True

    def delete_cell_mask(self, cell_n, frame=None, update_display=True):
        """
        Delete a cell mask from the current frame.

        Parameters
        ----------
        cell_n : int
            The cell number to delete.
        frame : Frame
            The frame to delete the cell from. If None, the current frame is used.
        update_display : bool
            Whether to update the display after deleting the cell.
        """

        if frame is None:
            frame = self.frame

        if frame == self.frame:  # remove the cell mask from the mask overlay
            self.canvas.add_cell_highlight(cell_n, color='none', layer='mask')

        self.stack.delete_cells([cell_n], frame_number=frame.frame_number)
        if hasattr(self.stack, 'tracked_centroids'):
            self.left_toolbar.also_save_tracking.setChecked(True)

        if update_display:
            print(f'Deleted cell {cell_n} from frame {frame.frame_number}')
            if frame == self.frame:
                self._update_tracking_overlay()
                self._update_display()
                self._update_ROIs_label()

    def save_tracking(self, event=None, file_path=None):
        """
        Save the tracking data to a CSV file.

        Parameters
        ----------
        file_path : str
            The file path to save the tracking data to. If None, a file dialog is opened.
        """

        if not self.file_loaded:
            return

        if not hasattr(self.stack, 'tracked_centroids'):
            print('No tracking data to save.')
            self.statusBar().showMessage('No tracking data to save.', 1500)
            return

        if file_path is None:
            file_path = QFileDialog.getSaveFileName(self, 'Save tracking data as...', filter='*.csv')[0]
            if file_path == '':
                return

        if hasattr(self.stack, 'mitoses'):
            mitoses_path = file_path.replace('tracking.csv', 'mitoses.csv')
            scores_path = file_path.replace('tracking.csv', 'mitosis_scores.csv')
            self.stack.mitoses.to_csv(mitoses_path)
            self.stack.mitosis_scores.to_csv(scores_path)
            print(f'Saved mitoses to {mitoses_path}')

        self.stack.save_tracking(file_path)
        print(f'Saved tracking data to {file_path}')

    def _load_tracking(self):
        if not self.file_loaded:
            return
        file_path = QFileDialog.getOpenFileName(self, 'Load tracking data...', filter='*.csv')[0]
        if file_path == '':
            return

        self.stack.load_tracking(file_path)
        print(f'Loaded tracking data from {file_path}')
        self.statusBar().showMessage(f'Loaded tracking data from {file_path}', 2000)
        self.left_toolbar.propagate_FUCCI_checkbox.setEnabled(True)
        self._recolor_tracks()

    def _save(self):
        if not self.file_loaded:
            return

        if self.left_toolbar.also_save_tracking.isChecked():
            self.save_tracking(file_path=self.stack.name + 'tracking.csv')

        if self.left_toolbar.save_stack.isChecked():
            frames_to_save = self.stack.frames

        else:
            frames_to_save = [self.stack.frames[self.frame_number]]

        for frame in self._progress_bar(frames_to_save):
            self.save_frame(frame)  # save the frame to the same file path

        print(f'Saved segmentation to {self.stack.name}')

    def _save_as(self):
        if not self.file_loaded:
            return

        if self.left_toolbar.save_stack.isChecked():
            folder_path = QFileDialog.getExistingDirectory(self, 'Save stack to folder...')
            if folder_path == '':
                return
            for frame in self._progress_bar(self.stack.frames):
                file_path = os.path.join(folder_path, os.path.basename(frame.name))
                self.save_frame(frame, file_path=file_path)
        else:
            file_path = QFileDialog.getSaveFileName(self, 'Save frame as...', filter='*_seg.npy')[0]
            folder_path = Path(file_path).parent
            if file_path == '':
                return
            if not file_path.endswith('_seg.npy'):
                file_path = file_path + '_seg.npy'
            self.save_frame(self.frame, file_path)

        print(f'Saved segmentation to {folder_path}')
        if self.left_toolbar.also_save_tracking.isChecked():
            self.save_tracking(file_path=folder_path + '/tracking.csv')

    def save_frame(self, frame, file_path=None):
        """
        Save a frame to a .seg.npy file.

        Parameters
        ----------
        frame : Frame
            The frame to save.
        file_path : str
            The file path to save the frame to. If None, the frame's name is used.
        """

        if file_path is None:
            file_path = frame.name
        if not frame.has_outlines:
            print('Generating outlines...')
            outlines_list = utils.outlines_list(frame.masks)
            for cell, outline in zip(frame.cells, outlines_list):
                cell.outline = outline
                cell.get_centroid()
            frame.has_outlines = True

        try:  # fetch cell cycle data if available
            self._convert_red_green([frame])
            write_attrs = ['cell_cycles']

        except AttributeError:
            write_attrs = []

        if hasattr(frame, 'zstack'):
            frame.img = frame.zstack[self.zstack_number]

        frame.to_seg_npy(file_path, write_attrs=write_attrs, overwrite_img=True)
        frame.name = file_path

    def _export_csv(self):
        if not self.file_loaded:
            return

        df = self._get_export()

        root_path = Path(self.stack.name) / 'export.csv'
        dialog = ExportWizard(df.iloc[:3], len(df), self, root_path.as_posix())
        if dialog.exec():
            # Retrieve data from export dialog
            save_path = dialog.save_path
            checked_attributes = dialog.checked_attributes
        else:
            return

        self.export_csv(save_path, columns=checked_attributes, csv_df=df)
        print(f'Saved CSV to {save_path}')

    def _get_export(self):
        for frame in self.stack.frames:
            if not frame.has_outlines:
                print(f'Generating outlines for frame {frame.frame_number}...')
                outlines_list = utils.outlines_list(frame.masks)
                for cell, outline in zip(frame.cells, outlines_list):
                    cell.outline = outline
                    cell.get_centroid()
                frame.has_outlines = True

        # get the data to export
        if hasattr(self.stack, 'tracked_centroids'):
            self.stack.get_velocities()
            df = self.stack.velocities.sort_values(['frame', 'cell_number'])
            if 'color' in df.columns:
                df.drop(columns='color', inplace=True)  # don't save this cosmetic column
        else:
            df = self.stack.centroids()

        self._convert_red_green()

        cells = np.concatenate([frame.cells for frame in self.stack.frames])
        attrs = self._get_cell_frame_attrs(ignored={'n', 'frame'})  # get all cell attributes except n and frame (redundant)

        for attr in attrs:
            df[attr] = np.array([getattr(cell, attr) for cell in cells])

        return df

    def export_csv(self, file_path, columns='all', csv_df=None):
        """
        Export all cell-level attributes to a CSV file.

        Parameters
        ----------
        file_path : str
            The file path to save the CSV file to.
        columns : list
            A list of column names to export. If 'all', all columns are exported.
        csv_df : pd.DataFrame
            The DataFrame to export. If None, the export DataFrame is generated.
        """
        for frame in self.stack.frames:
            if not frame.has_outlines:
                print(f'Generating outlines for frame {frame.frame_number}...')
                outlines_list = utils.outlines_list(frame.masks)
                for cell, outline in zip(frame.cells, outlines_list):
                    cell.outline = outline
                    cell.get_centroid()
                frame.has_outlines = True

        if csv_df is None:
            csv_df = self._get_export()

        if columns == 'all':
            export = csv_df
        else:
            export = csv_df[columns]

        export.to_csv(file_path, index=False)

    def _convert_red_green(self, frames=None):
        """convert cell.red, cell.green attributes to FUCCI labeling for the stack."""
        if frames is None:
            frames = self.stack.frames
        for frame in frames:
            try:
                green, red = np.array(frame.get_cell_attrs(['green', 'red'])).T
            except AttributeError:
                self._get_red_green(frame)
                green, red = np.array(frame.get_cell_attrs(['green', 'red'])).T
            except ValueError:  # no cells in the frame
                continue
            frame.cell_cycles = green + 2 * red
            frame.set_cell_attrs('cycle_stage', frame.cell_cycles)

    def _FUCCI_overlay_changed(self):
        if not self.file_loaded:
            return

        self.left_toolbar.tabbed_widget.blockSignals(True)  # manually switch tabs (without triggering tab switch event)
        self.left_toolbar.tabbed_widget.setCurrentIndex(1)  # switch to the FUCCI tab
        self.current_tab = 1
        self.left_toolbar.tabbed_widget.blockSignals(False)
        overlay_color = self.FUCCI_dropdown

        # set RGB mode
        if overlay_color == 0:
            self.left_toolbar.RGB_visible = True
        else:
            if overlay_color == 3:
                self.left_toolbar.RGB_visible = [True, True, False]
            elif overlay_color == 2:
                self.left_toolbar.RGB_visible = [True, False, False]
            elif overlay_color == 1:
                self.left_toolbar.RGB_visible = [False, True, False]

            # set overlay mode
            self.outlines_visible = True
            self.masks_visible = False

        self._FUCCI_overlay()

    def _FUCCI_overlay(self, event=None):
        """Handle cell cycle overlay options."""
        FUCCI_index = self.FUCCI_dropdown
        overlay_color = ['none', 'g', 'r', 'orange'][FUCCI_index]
        if self.left_toolbar.tabbed_widget.currentIndex() != 1 or FUCCI_index == 0:
            self.canvas.clear_overlay('FUCCI')  # clear FUCCI overlay during basic selection
            self.FUCCI_mode = False
            return

        else:
            self.select_cell(None)
            self.FUCCI_mode = True
            self.canvas.clear_overlay('selection')  # clear basic selection during FUCCI labeling
            if len(self.frame.cells) == 0:
                return
            if FUCCI_index == 3:
                colors = np.array(['g', 'r', 'orange'])
                green, red = np.array(self.frame.get_cell_attrs(['green', 'red'])).T
                colored_cells = np.where(red | green)[0]  # cells that are either red or green
                cell_cycle = green + 2 * red - 1
                cell_colors = colors[cell_cycle[colored_cells]]  # map cell cycle state to green, red, orange
                self.canvas.highlight_cells(colored_cells, alpha=1, cell_colors=cell_colors, img_type='outlines', layer='FUCCI')

            else:
                colored_cells = np.where(self.frame.get_cell_attrs(overlay_color))[0]
                self.canvas.highlight_cells(colored_cells, alpha=1, color=overlay_color, img_type='outlines', layer='FUCCI')

    def reset_display(self):
        """
        Reset the display to its initialized state.
        """
        self.drawing_cell_roi = False
        self.drawing_cell_split = False
        self.select_cell(None)
        self.FUCCI_dropdown = 0  # clear FUCCI overlay
        self.canvas.clear_overlay('tracking')
        self.canvas.clear_overlay('mitosis')
        self.seg_overlay_attr.setCurrentIndex(0)  # clear attribute overlay
        self.left_toolbar.RGB_visible = True
        if not self.is_grayscale:
            self.left_toolbar.show_grayscale_checkbox.setChecked(False)
        self.canvas.clear_overlay('selection')  # remove any selection overlays (highlighting, outlines)
        self.canvas.img_plot.autoRange()

        self.histogram_menu.setCurrentIndex(0)
        self.time_series_menu.setCurrentIndex(0)
        self.particle_stat_menu.setCurrentIndex(0)

    def imshow(self):
        """Render any changes to the image data (new file, new frame, new z slice)."""
        execution_times = {}

        start_time = time.time()
        self._update_tracking_overlay()
        execution_times['update_tracking_overlay'] = time.time() - start_time

        start_time = time.time()
        self._update_ROIs_label()
        execution_times['update_ROIs_label'] = time.time() - start_time

        self._update_display()

        start_time = time.time()
        self._show_seg_overlay()
        execution_times['show_seg_overlay'] = time.time() - start_time

        start_time = time.time()
        self._plot_histogram()
        execution_times['histogram_plot'] = time.time() - start_time

        if debug_execution_times:
            print('-----------IMSHOW TIMES-----------')
            sorted_execution_times = sorted(execution_times.items(), key=lambda item: item[1], reverse=True)
            for description, duration in sorted_execution_times:
                if duration < 0.001:
                    continue
                print(f'{description}: {duration:.4f} seconds')

    def keyPressEvent(self, event):
        """Handle key press events (e.g., arrow keys for frame navigation)."""
        if event.key() == Qt.Key.Key_Escape:
            if self.is_iterating:  # cancel progress bar iteration
                self.cancel_iter = True

        if event.key() == Qt.Key.Key_Tab:
            # switch between tabs
            if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                current_tab = self.left_toolbar.tabbed_widget.currentIndex()
                self.left_toolbar.tabbed_widget.setCurrentIndex((current_tab + 1) % self.left_toolbar.tabbed_widget.count())

        if not self.file_loaded:
            return

        # Enter key completes cell drawing, mitosis mode
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.drawing_cell_roi:
                self._close_cell_roi()
            elif self.drawing_cell_split:
                self._split_cell()
            elif self.mitosis_mode:
                self._mitosis_selected(None)  # skip this prompt, advance to next
            return

        # Ctrl-key shortcuts
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if not self.is_grayscale:
                # FUCCI labeling modes
                if event.key() == Qt.Key.Key_R:
                    if self.FUCCI_dropdown == 2 and self.FUCCI_mode:
                        self.FUCCI_dropdown = 0
                        self.left_toolbar.RGB_visible = True
                    else:
                        self.left_toolbar.tabbed_widget.setCurrentIndex(1)
                        self.FUCCI_dropdown = 2
                        self.left_toolbar.RGB_visible = [True, False, False]
                    return

                elif event.key() == Qt.Key.Key_G:
                    if self.FUCCI_dropdown == 1 and self.FUCCI_mode:
                        self.FUCCI_dropdown = 0
                        self.left_toolbar.RGB_visible = True
                    else:
                        self.left_toolbar.tabbed_widget.setCurrentIndex(1)
                        self.FUCCI_dropdown = 1
                        self.left_toolbar.RGB_visible = [False, True, False]
                    return

                elif event.key() == Qt.Key.Key_A:
                    if self.FUCCI_dropdown == 3 and self.FUCCI_mode:
                        self.FUCCI_dropdown = 0
                        self.left_toolbar.RGB_visible = True
                    else:
                        self.left_toolbar.tabbed_widget.setCurrentIndex(1)
                        self.FUCCI_dropdown = 3
                        self.left_toolbar.RGB_visible = [True, True, False]
                    return

        # r-g-b toggles
        if not self.is_grayscale:
            if event.key() == Qt.Key.Key_R:
                self.left_toolbar.RGB_checkboxes[0].toggle()
            elif event.key() == Qt.Key.Key_G:
                self.left_toolbar.RGB_checkboxes[1].toggle()
            elif event.key() == Qt.Key.Key_B:
                self.left_toolbar.RGB_checkboxes[2].toggle()

        # segmentation overlay
        if event.key() == Qt.Key.Key_X:
            self.masks_visible = not self.masks_visible
        elif event.key() == Qt.Key.Key_Z:
            self.outlines_visible = not self.outlines_visible
        elif event.key() == Qt.Key.Key_Delete:
            if self.selected_cell_n is not None:
                self.delete_cell_mask(self.selected_cell_n)
                self.select_cell(None)

        # cancel drawing
        if event.key() == Qt.Key.Key_Escape:
            if self.drawing_cell_roi:
                self.cell_roi.clearPoints()
                self.drawing_cell_roi = False
            if self.drawing_cell_split:
                self.cell_split.clearPoints()
                self.drawing_cell_split = False
            if self.mitosis_mode:
                self._cancel_mitosis()
            else:  # deselect cell
                self.select_cell(None)

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

    def _reset_view(self):
        """Reset the view to the original image data."""
        self.FUCCI_dropdown = 0
        self.left_toolbar.RGB_visible = True
        self.canvas.img_plot.autoRange()
        if not self.is_grayscale:
            self.left_toolbar.show_grayscale_checkbox.setChecked(False)

    def _toggle_grayscale(self):
        self.left_toolbar.show_grayscale_checkbox.toggle()

    def _toggle_inverted(self):
        self.left_toolbar.inverted_checkbox.toggle()

    def _show_grayscale_toggled(self, event=None):
        if event is None:
            event = self.left_toolbar.show_grayscale_checkbox.isChecked()
        self.canvas.img.set_grayscale(event != 0)

    def _reorder_channels(self):
        if not self.file_loaded:
            return
        if self.is_grayscale:
            return
        from .qt import ChannelOrderDialog

        saved_img = self.canvas.img.img_data
        saved_levels = self.canvas.img.getLevels()

        def preview_reorder(channel_order):
            """Preview the current image with the given channel order."""
            new_img = self.frame.img[..., channel_order]
            new_levels = [saved_levels[i] for i in channel_order]
            self.canvas.img.setImage(new_img)
            self.canvas.img.setLevels(new_levels)

        def clear_preview():
            """Clear the preview of the current image."""
            self.canvas.img.setImage(saved_img)
            self.canvas.img.setLevels(saved_levels)

        def finish_reordering(channel_order):
            """Apply the channel order to the entire stack."""
            if self.reorder_dialog.result() == QDialog.DialogCode.Accepted:
                self.reorder_channels(channel_order)

        # Prompt the user for the channel order
        self.reorder_dialog = ChannelOrderDialog(self)
        self.reorder_dialog.previewRequested.connect(preview_reorder)
        self.reorder_dialog.clearPreviewRequested.connect(clear_preview)
        self.reorder_dialog.finished.connect(finish_reordering)

        self.reorder_dialog.exec()

    def reorder_channels(self, channel_order):
        if len(channel_order) != 3:
            raise ValueError('Channel order must have exactly three elements.')

        for frame in self._progress_bar(self.stack.frames, desc='Reordering channels'):
            frame.img = frame.img[..., channel_order]

            if hasattr(frame, 'zstack'):
                frame.zstack = frame.zstack[..., channel_order]
            if hasattr(frame, 'bounds'):
                frame.bounds = frame.bounds[..., channel_order, :]

        self._autorange_LUT_sliders()
        self._update_display()

    def _change_LUTs(self):
        from .qt import LookupTableDialog

        LUT_options = list(self.canvas.img.LUT_options.keys())
        initial_LUTs = self.canvas.img.LUTs

        def apply_LUTs(new_LUTs):
            self.canvas.img.LUTs = new_LUTs
            self.canvas.img.update_LUTs()

        def revert_LUTs():
            self.canvas.img.LUTs = initial_LUTs
            self.canvas.img.update_LUTs()

        dialog = LookupTableDialog(self, options=LUT_options, initial_LUTs=initial_LUTs)
        dialog.valueChanged.connect(apply_LUTs)
        dialog.rejected.connect(revert_LUTs)
        dialog.exec()

    def _clear_stored_overlays(self):
        for frame in self.stack.frames:
            if hasattr(frame, 'stored_mask_overlay'):
                del frame.stored_mask_overlay
        self.canvas.draw_masks_parallel()

    def rotate_clockwise(self):
        """
        Rotate the stack 90 degrees clockwise.
        """
        self.stack.rot90(k=1, progress=self._progress_bar)
        self._clear_stored_overlays()
        self._update_display()
        self.canvas.img_plot.autoRange()

    def rotate_counterclockwise(self):
        """
        Rotate the stack 90 degrees counterclockwise.
        """
        self.stack.rot90(k=3, progress=self._progress_bar)
        self._clear_stored_overlays()
        self._update_display()
        self.canvas.img_plot.autoRange()

    # Drag and drop event
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        # TODO: figure out how to release the dragged window
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        self.open_stack(natsorted(files))

    def _scan_tracked_centroids(self):
        if not hasattr(self.stack, 'tracked_centroids'):
            return
        t = self.stack.tracked_centroids
        for frame in self.stack.frames:
            tracked_frame = t[t.frame == frame.frame_number]
            tracked_cells = tracked_frame['cell_number']
            frame_cells = frame.get_cell_attrs('n')

            missing_tracks = set(frame_cells) - set(tracked_cells)
            if len(missing_tracks) > 0:
                QMessageBox.warning(
                    self,
                    'Missing tracks',
                    f'tracked_centroids is missing {len(missing_tracks)} cells in frame {frame.frame_number}: {missing_tracks}',
                )
                print(f'tracked_centroids is missing {len(missing_tracks)} cells in frame {frame.frame_number}: {missing_tracks}')

            extra_tracks = set(tracked_cells) - set(frame_cells)
            if len(extra_tracks) > 0:
                QMessageBox.warning(
                    self,
                    'Extra tracks',
                    f'tracked_centroids has {len(extra_tracks)} extra tracks in frame {frame.frame_number}: {extra_tracks}',
                )
                print(f'tracked_centroids has {len(extra_tracks)} extra tracks in frame {frame.frame_number}: {extra_tracks}')

    # --------------I/O----------------
    def take_screenshot(self):
        """
        Take a screenshot of the main window.

        Returns
        -------
        screenshot : QImage
            The screenshot as a QImage.
        """
        screen = QApplication.primaryScreen()
        if screen:
            screenshot = screen.grabWindow(self.winId())  # Capture the window
            return screenshot
        else:
            return None

    def save_screenshot(self, file_path=None):
        """
        Take a screenshot of the main window and save it to a file.

        Parameters
        ----------
        file_path : str
            The file path to save the screenshot to. If None, a file dialog is opened.
        """

        if file_path is None:
            file_path = QFileDialog.getSaveFileName(self, 'Save screenshot as...', filter='*.png')[0]
            if file_path == '':
                return
        screenshot = self.take_screenshot()
        if screenshot is None:
            return
        screenshot.save(file_path, 'png')  # Save to file
        print(f'Saved screenshot to {file_path}')

    def _save_stack_gif(self):
        if not self.file_loaded:
            return

        file_path = QFileDialog.getSaveFileName(self, 'Save stack as GIF...', filter='*.gif')[0]
        if file_path == '':
            return

        delay = 100  # ms between frames
        try:
            self.save_stack_gif(file_path, delay=delay)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save GIF: {str(e)}')

    def save_stack_gif(self, file_path, delay=100):
        """
        Save the stack with its current visual settings as an animated GIF.

        Parameters
        ----------
        file_path : str
            The file path to save the GIF to.
        delay : int
            The delay between frames in milliseconds.
        """

        import io

        from PIL import Image
        from PyQt6.QtCore import QBuffer, QByteArray, QTimer
        from PyQt6.QtWidgets import QApplication

        images = []

        # Ensure main window is active and process events
        self.activateWindow()
        self.raise_()

        # Process events and give a small delay for window activation
        QApplication.processEvents()
        QTimer.singleShot(100, lambda: None)  # Wait for 100ms

        # Convert each frame to PIL Image
        for frame_number in tqdm(range(len(self.stack.frames)), desc='Saving GIF'):
            self.change_current_frame(frame_number)
            QApplication.processEvents()  # Allow GUI updates
            # Convert QImage to PIL Image
            qimage = self.take_screenshot().toImage()

            # Convert QImage to bytes using QByteArray
            byte_array = QByteArray()
            buffer = QBuffer(byte_array)
            buffer.open(QBuffer.OpenModeFlag.WriteOnly)
            qimage.save(buffer, 'PNG')

            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(byte_array.data()))
            images.append(pil_image.convert('RGBA'))

        # Save as animated GIF
        if images:
            images[0].save(file_path, save_all=True, append_images=images[1:], duration=delay, loop=0, optimize=True)

    def open_stack(self, files):
        """
        Open a stack of images or segmentation files.
        If multiple file types are present, the function will attempt to load the segmented files first, then the image files.

        Parameters
        ----------
        files : str or list
            The file path(s) to open. If a folder is selected, all files in the folder will be loaded (if valid).
        """

        if isinstance(files, str):
            files = [files]

        loaded_stack = self._load_files(files)
        if not loaded_stack:
            return

        self.stack = loaded_stack
        self.globals_dict['stack'] = self.stack

        self.file_loaded = True
        if hasattr(self.stack, 'tracked_centroids'):
            self.left_toolbar.propagate_FUCCI_checkbox.setEnabled(True)
            self._recolor_tracks()
        else:
            self.left_toolbar.propagate_FUCCI_checkbox.setChecked(False)
            self.left_toolbar.propagate_FUCCI_checkbox.setEnabled(False)

        if len(self.stack.frames) == 1:
            out_message = f'Loaded frame {self.stack.frames[0].name}.'
        else:
            out_message = f'Loaded stack {self.stack.name} with {len(self.stack.frames)} frames.'
        print(out_message)
        self.statusBar().showMessage(out_message, 3000)

        self.frame = self.stack.frames[0]
        if hasattr(self.frame, 'zstack'):
            self.is_zstack = True
            self.zstack_slider.setVisible(True)
            self.zstack_slider.setRange(0, self.frame.zstack.shape[0] - 1)
            self.zstack_slider.setValue(0)
            self.zstack_number = 0
        else:
            self.is_zstack = False
            self.zstack_slider.setVisible(False)

        if self.frame.img.ndim == 2:  # single channel
            self.left_toolbar.grayscale_mode()
        elif self.frame.img.ndim == 3:  # RGB
            self.left_toolbar.RGB_mode()
        else:
            raise ValueError(f'{self.frame.name} has {self.frame.img.ndim} image dimensions, must be 2 (grayscale) or 3 (RGB).')

        if len(self.stack.frames) > 1:
            self.frame_slider.setVisible(True)
        else:
            self.frame_slider.setVisible(False)

        self.frame_slider.setRange(0, len(self.stack.frames) - 1)
        self._autorange_LUT_sliders()
        self.change_current_frame(0, reset=True)  # call frame update explicitly (in case the slider value was already at 0)

        for frame in self.stack.frames:  # loaded segmentation files will always have outlines
            frame.has_outlines = True

        # reset visual settings
        self.canvas.img_plot.autoRange()

        self.left_toolbar.saved_visual_settings = [self.default_visual_settings for _ in range(4)]

        self._precompute_bounds()
        self.canvas.draw_masks_parallel()

    def _open_files(self):
        files = QFileDialog.getOpenFileNames(self, 'Open file(s)', filter='*seg.npy *.tif *.tiff *.nd2')[0]
        if len(files) > 0:
            self.open_stack(files)

    def _open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Open folder of segmentation files')
        if folder:
            self.open_stack([folder])

    def _load_files(self, files):
        """
        Load a stack of images.
        If a tracking.csv is found, the tracking data is returned as well
        """
        tracking_file = None

        # ----figure out what's being loaded----
        if os.path.isdir(files[0]):  # if a folder is selected, load all files in the folder
            seg_files = []
            img_files = []

            for f in natsorted(os.listdir(files[0])):
                if f.endswith('seg.npy'):
                    seg_files.append(os.path.join(files[0], f))
                elif f.lower().endswith('tif') or f.lower().endswith('tiff') or f.lower().endswith('nd2'):
                    img_files.append(os.path.join(files[0], f))
                elif f.endswith('tracking.csv'):
                    tracking_file = os.path.join(files[0], f)

        else:  # list of files
            seg_files = [f for f in files if f.endswith('seg.npy')]
            img_files = [f for f in files if f.lower().endswith('tif') or f.lower().endswith('tiff') or f.lower().endswith('nd2')]
            tracking_files = [f for f in files if f.endswith('tracking.csv')]
            if len(tracking_files) > 0:
                tracking_file = tracking_files[-1]

        # ----load the files----
        # only loads one type of file per call
        # tries to load seg.npy files first, then image files
        if len(seg_files) > 0:  # segmented file paths
            stack = SegmentedStack(frame_paths=seg_files, load_img=True, progress_bar=self._progress_bar)
            if tracking_file is not None:
                stack.load_tracking(tracking_file)
                if os.path.exists(tracking_file.replace('tracking.csv', 'mitoses.csv')):
                    try:
                        stack.mitoses = pd.read_csv(tracking_file.replace('tracking.csv', 'mitoses.csv'), index_col=0)
                        stack.mitosis_scores = pd.read_csv(
                            tracking_file.replace('tracking.csv', 'mitosis_scores.csv'), index_col=0
                        )
                        print(f'Loaded mitoses from {tracking_file.replace("tracking.csv", "mitoses.csv")}')
                    except FileNotFoundError:
                        print(f'Failed to load mitoses from {tracking_file.replace("tracking.csv", "mitoses.csv")}')
                        if hasattr(stack, 'mitoses'):
                            del stack.mitoses

            self.file_loaded = True
            return stack

        elif len(img_files) > 0:  # image files
            from segmentation_viewer.io import read_image_file

            frames = []
            for file_path in img_files:
                file_path = Path(file_path)
                imgs = read_image_file(str(file_path), progress_bar=self._progress_bar, desc=f'Loading {file_path.name}')
                if imgs is None:
                    return False, None
                for v, img in enumerate(self._progress_bar(imgs, desc=f'Processing {file_path.stem}')):
                    if img.shape[-1] == 2:  # pad to 3 color channels
                        img = np.stack([img[..., 0], img[..., 1], np.zeros_like(img[..., 0])], axis=-1)
                    elif img.shape[-1] == 1:  # single channel
                        img = img[..., 0]  # drop the last dimension

                    if len(img) > 1:  # z-stack
                        frames.append(segmentation_from_zstack(img, name=file_path.with_name(file_path.stem + f'-{v}_seg.npy')))
                    else:  # single slice
                        frames.append(segmentation_from_img(img[0], name=file_path.with_name(file_path.stem + f'-{v}_seg.npy')))

            stack = SegmentedStack(from_frames=frames)
            self.file_loaded = True
            return stack

        else:  # can't find any seg.npy or tiff files, ignore
            self.statusBar().showMessage(f'ERROR: File {files[0]} is not a seg.npy or tiff file, cannot be loaded.', 4000)
            return False

    def delete_frame(self, event=None, frame_number=None):
        """
        Delete a frame from the stack.

        Parameters
        ----------
        frame_number : int
            The frame number to delete. If None, the current frame is deleted.
        """

        if not self.file_loaded:
            return

        if frame_number is None:
            frame_number = self.frame_number

        if len(self.stack.frames) == 1:
            return

        self.stack.delete_frame(frame_number)
        self.frame_slider.setRange(0, len(self.stack.frames) - 1)
        self.change_current_frame(min(frame_number, len(self.stack.frames) - 1))

    def make_substack(self, substack_frames=None):
        """
        Create a substack from the current stack.

        Parameters
        ----------
        substack_frames : list
            A list of frame numbers to include in the substack. If None, a dialog is opened to select the frames.
        """

        if not self.file_loaded:
            return

        if substack_frames is None:
            # popup dialog to select the range of frames to include in the substack
            dialog = SubstackDialog(len(self.stack.frames), self)
            if dialog.exec() == QDialog.Accepted:
                substack_frames = dialog.get_input()
                if substack_frames is None:
                    return

        substack_frames = np.array(substack_frames)
        self.stack.make_substack(substack_frames)
        self.frame_slider.setRange(0, len(self.stack.frames) - 1)
        self.change_current_frame(min(self.frame_number, len(self.stack.frames) - 1))

    # def import_masks(self):

    def import_images(self, event=None, files=None, image_shape=None):
        """
        Import images to the current stack.

        Parameters
        ----------
        files : str or list
            The file path(s) to import. If None, a file dialog is opened.
        image_shape : tuple or str
            The shape of the images to import.
            If 'all', all images are imported in full.
            If tuple, treated as (t_bounds, p_bounds, z_bounds, c_bounds), where each bound is a slice or numpy index array.
            if None, a shape dialog is opened.
        """

        from segmentation_viewer.io import read_image_file

        if not self.file_loaded:
            return

        if files is None:
            files = natsorted(QFileDialog.getOpenFileNames(self, 'Open image file(s)', filter='*.tif *.tiff *.nd2')[0])
        elif isinstance(files, str):
            files = [files]

        if len(files) == 0:
            return

        imgs = []
        for file in files:
            name = Path(file).name
            file_imgs = read_image_file(
                file, progress_bar=self._progress_bar, desc=f'Importing images from {name}', image_shape=image_shape
            )
            if file_imgs is None:
                continue
            imgs.extend(file_imgs)

        if len(imgs) == 0:  # no images loaded
            return

        # TODO: check if the number of images matches the number of frames in the stack
        for img, frame in zip(imgs, self.stack.frames):
            if img.shape[-1] == 1:  # single channel
                img = img[..., 0]  # drop the last dimension
                self.left_toolbar.grayscale_mode()
            else:  # RGB
                if img.shape[-1] == 2:  # pad to 3 color channels
                    img = np.stack([img[..., 0], img[..., 1], np.zeros_like(img[..., 0])], axis=-1)
                self.left_toolbar.RGB_mode()

            frame.img = img[0]
            if len(img) > 1:  # z-stack
                self.zstack_number = 0  # if any z-stack images are loaded, reset the z-stack number (a little redundant)
                frame.zstack = img
            else:  # single slice
                if hasattr(frame, 'zstack'):
                    del frame.zstack

            if hasattr(frame, 'bounds'):
                del frame.bounds

        self.change_current_frame(self.frame_number)

    def _get_red_green(self, frame=None):
        """Fetch or create red and green attributes for cells in the current frame."""
        if frame is None:
            frame = self.frame

        for cell in frame.cells:
            if hasattr(cell, 'cycle_stage'):
                cell.green = cell.cycle_stage == 1 or cell.cycle_stage == 3
                cell.red = cell.cycle_stage == 2 or cell.cycle_stage == 3
            else:
                cell.red = False
                cell.green = False

    def _update_packages(self):
        """
        Update segmentation_tools and segmentation_viewer packages from GitHub.
        Pulls only the src directories and updates the local installations.
        """
        from segmentation_viewer.update import update_packages

        try:
            update_packages()
        except Exception as e:
            QMessageBox.critical(self, 'Update Failed', f'Package update error: {str(e)}')
            return
        QMessageBox.information(self, 'Update Complete', 'Packages updated successfully. Please restart the application.')

    def closeEvent(self, event):
        # Close the command line window when the main window is closed
        # find all attrs that end with "dialog" or "window"
        for attr in dir(self):
            if attr.endswith('dialog') or attr.endswith('window'):
                try:
                    getattr(self, attr).close()
                except Exception:
                    pass

        self._dump_config()
        self.canvas.close()
        event.accept()


def main():
    pg.setConfigOptions(useOpenGL=True)
    # pg.setConfigOptions(enableExperimental=True)

    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()

    # apply dark theme
    darktheme_stylesheet = load_stylesheet(importlib.resources.files('segmentation_viewer.assets').joinpath('darktheme.qss'))
    app.setStyleSheet(darktheme_stylesheet)

    app.quitOnLastWindowClosed = True
    ui = MainWidget()
    ui.show()
    app.exec()


if __name__ == '__main__':
    main()
