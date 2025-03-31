import importlib.resources
import os
import sys
import time
from multiprocessing import cpu_count
from pathlib import Path

import fastremap
import numpy as np
import pandas as pd
import pyqtgraph as pg
from natsort import natsorted
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFontMetrics, QIcon, QIntValidator, QUndoCommand
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
from segmentation_tools.io import segmentation_from_img, segmentation_from_zstack
from segmentation_tools.preprocessing import get_quantile
from segmentation_tools.segmented_comprehension import Cell, SegmentedImage, SegmentedStack
from segmentation_tools.utils import cell_scalar_attrs, frame_array_attrs, masks_to_outlines, outlines_list
from skimage import draw
from tqdm import tqdm

from .canvas import CellMaskPolygons, CellSplitLines, PyQtGraphCanvas
from .command_line import CommandLineWindow
from .io import ExportWizard
from .qt import CustomComboBox, FrameStackDialog, QueuedUndoStack, SubstackDialog, UndoHistoryWindow, labeled_LUT_slider
from .scripting import ScriptWindow
from .ui import LeftToolbar, calculate_range_params, clear_layout
from .utils import create_html_table, load_stylesheet
from .workers import BoundsProcessor

# high priority
# TODO: generalized data analysis pipeline. Ability to identify any img-shaped attributes in the frame and overlay them a la heights
# ndimage labeled measurements on any of these attributes to create new ones
# TODO: frame histogram should have options for aggregating over frame or stack
# TODO: import masks (and everything else except img/zstack)

# low priority
# TODO: unify print statements with status bar messages
# TODO: when cell is clicked, have option to show its entire colormapped track
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
        self.undo_stack = QueuedUndoStack(self)
        self.globals_dict['history'] = self.undo_stack
        self.model_type = 'cyto3'
        self.progress_widget = None

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
        self.file_loaded = False

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
        self.undo_window = None

        self._load_config()
        self._apply_overlay_settings()

        # ----------------Connections----------------
        self.frame_slider.valueChanged.connect(self.change_current_frame)
        self.zstack_slider.valueChanged.connect(self.update_zstack_number)

        # click event
        self.canvas.img_plot.scene().sigMouseClicked.connect(self._on_click)
        self.canvas.seg_plot.scene().sigMouseClicked.connect(self._on_click)

    @property
    def open_dir(self) -> str:
        if not hasattr(self, '_open_dir'):
            self._open_dir = Path.cwd()
        return str(self._open_dir)

    @open_dir.setter
    def open_dir(self, path):
        self._open_dir = path

    @property
    def save_dir(self) -> str:
        if not hasattr(self, '_save_dir'):
            self._save_dir = self.open_dir
        return str(self._save_dir)

    @save_dir.setter
    def save_dir(self, path):
        self._save_dir = path

    def _init_actions(self):
        from .utils import create_action

        file_actions = {
            'Open File(s)': (self._open_files, 'Ctrl+O'),
            'Save': (self._save, 'Ctrl+S'),
            'Save As...': (self._save_as, 'Ctrl+Shift+S'),
            'Export': {
                'Export CSV...': (self._export_csv, 'Ctrl+Shift+E'),
                'Export Heights...': (self._export_heights, None),
                'Export ImageJ ROIs...': (self._export_ROIs, None),
            },
            'Import': {'Import Images...': (self.import_images, None), 'Import Heights...': (self._import_heights, None)},
            'Save Screenshot': {
                'Window': (self._save_screenshot, None),
                'Image Plot': (self._save_img_plot, None),
                'Segmentation Plot': (self._save_seg_plot, None),
            },
            'Save Video': {
                'Window': (self._export_window_video, None),
                'Image Plot': (self._export_img_plot_video, None),
                'Segmentation Plot': (self._export_seg_plot_video, None),
            },
            'Exit': (self.close, 'Ctrl+Q'),
        }
        edit_actions = {
            'Undo': (self.undo_stack.undo, 'Ctrl+Z'),
            'Redo': (self.undo_stack.redo, ('Ctrl+Shift+Z', 'Ctrl+Y')),
            'Undo History': (self._show_undo_history, None),
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
            'Change Image LUTs...': (self._change_LUTs, None),
            'Change Overlay Colormap...': (self._change_overlay_colormap, None),
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
        def create_menu(menu, items):
            for key, item in items.items():
                if isinstance(item, dict):
                    submenu = menu.addMenu(key)
                    create_menu(submenu, item)
                else:
                    function, shortcut = item
                    action = create_action(key, function, self, shortcut)
                    setattr(self, f'{function.__name__}_action', action)
                    menu.addAction(action)

        for menu, actions in zip(
            ['File', 'Edit', 'View', 'Image', 'Stack', 'Scripts', 'Help'],
            [file_actions, edit_actions, view_actions, image_actions, stack_actions, scripts_actions, help_actions],
        ):
            current_menu = self.menuBar().addMenu(menu)
            create_menu(current_menu, actions)

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
        if 'LUTs' in config:
            self.canvas.img.LUTs = tuple(config['LUTs'])
        if 'overlay_settings' in config:
            self.canvas.dark_overlay_settings = config['overlay_settings']
        if 'inverted_overlay_settings' in config:
            self.canvas.light_overlay_settings = config['inverted_overlay_settings']
        if 'inverted' in config:
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
        LUTs = list(self.canvas.img.LUTs)
        overlay_settings = getattr(self.canvas, 'dark_overlay_settings', current_overlay_settings.copy())
        inverted_overlay_settings = getattr(self.canvas, 'light_overlay_settings', current_overlay_settings.copy())

        config = {
            'overlay_settings': overlay_settings,
            'inverted_overlay_settings': inverted_overlay_settings,
            'inverted': inverted,
            'LUTs': LUTs,
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
        self.stat_LUT_layout = QVBoxLayout()
        self.stat_LUT_layout.setContentsMargins(0, 0, 0, 0)
        self.get_stat_LUT_slider()

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
        stat_overlay_layout.addLayout(self.stat_LUT_layout)

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
        self.seg_overlay_attr.currentIndexChanged.connect(self._new_seg_overlay)
        self.stat_frame_button.toggled.connect(self._update_stat_LUT)
        self.stat_stack_button.toggled.connect(self._update_stat_LUT)
        self.stat_custom_button.toggled.connect(self._update_stat_LUT)
        return right_scroll_area

    def get_stat_LUT_slider(self, mode='int'):
        clear_layout(self.stat_LUT_layout)
        slider_layout, self.stat_LUT_slider, self.stat_range_labels = labeled_LUT_slider(
            default_range=(0, 255), parent=None, digit_width=8, mode=mode
        )
        self.stat_LUT_layout.addLayout(slider_layout)
        self.stat_LUT_slider.valueChanged.connect(self._stat_LUT_slider_changed)

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
        buttons = [self.stat_frame_button, self.stat_stack_button, self.stat_custom_button]
        selected = [button.isChecked() for button in buttons].index(True)
        self.stat_LUT_type = ['frame', 'stack', 'custom'][selected]

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
        ignored_attrs = {'cycle_stage', 'n', 'frame', 'red', 'green', 'scale'}
        attrs = cell_scalar_attrs(cell) - ignored_attrs

        return attrs

    def _get_frame_array_attrs(self, frame=None, ignored={'masks', 'outlines'}):
        """Return all attributes which are image-shaped arrays"""
        if not self.file_loaded:
            return []

        if frame is None:
            frame = self.frame

        keys = set(frame_array_attrs(frame))
        return list(keys-ignored)

    def _get_cell_frame_attrs(self, ignored={'frame', 'n', 'green', 'red', 'scale'}):
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
        self._cell_attrs = self._get_cell_frame_attrs(ignored={'frame', 'n', 'green', 'red', 'cycle_stage', 'scale'})
        self._frame_attrs = self._get_frame_array_attrs(self.frame, ignored={'masks', 'outlines', 'scaled_heights'})
        keys = self._cell_attrs + self._frame_attrs

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
        if not self.file_loaded:
            return
        plot_attr = self.seg_overlay_attr.currentText()

        if plot_attr == 'Select Cell Attribute' or plot_attr is None:
            self.stat_LUT_slider.blockSignals(True)
            self.stat_LUT_slider.setRange(0, 255)
            self.stat_LUT_slider.setValue((0, 255))
            self.stat_LUT_slider.blockSignals(False)

        else:
            if plot_attr in self._frame_attrs:
                stat = []
                for frame in self.stack.frames:
                    val = getattr(frame, plot_attr, None)
                    if val is not None:
                        stat.append(val.flatten())
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

            min_val, max_val, step = calculate_range_params(stat)
            self.stat_stack_bounds = tuple(get_quantile(stat[..., np.newaxis])[0])
            if isinstance(step, (int, np.integer)):
                mode = 'int'
            else:
                mode = 'float'
            self.get_stat_LUT_slider(mode=mode)
            self.stat_LUT_slider.blockSignals(True)
            self.stat_LUT_slider.rescale(min_val, max_val, step)
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
            if plot_attr in self._frame_attrs:
                if not hasattr(self.frame, plot_attr):
                    return
                if plot_attr == 'heights':
                    if not hasattr(self.frame, 'z_scale'):
                        print(f'No z scale found for {self.frame.name}, defaulting to 1.')
                        self.left_toolbar.z_size = 1.0

                    plot_attr='scaled_heights'
                
                array_values = getattr(self.frame, plot_attr)
                self._overlay_seg_stat(array_values)
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
            if self.stat_LUT_type == 'frame':
                stat_range = tuple(get_quantile(stat[..., np.newaxis])[0])
                levels = stat_range
            elif self.stat_LUT_type == 'stack':
                levels = self.stat_stack_bounds
            elif self.stat_LUT_type == 'custom':
                levels = self.stat_LUT_slider.value()
            self.canvas.seg_stat_overlay.setImage(self.canvas.image_transform(stat))

        if self.stat_LUT_slider.__class__.__name__ == 'FineRangeSlider':  # integer slider
            levels = int(levels[0]), int(levels[1])

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
        self._refresh_segmentation()

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

        self._update_tracking_overlay()
        self._refresh_segmentation()
        self.masks_visible = True

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
        self.circle_mask = QGraphicsEllipseItem(padding, img_shape[0] + padding, diameter, diameter)
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

            self.cellpose_model = models.CellposeModel(gpu=True, model_type=self.model_type)
            self.size_model_path = models.size_model_path(self.model_type)
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
        self._refresh_segmentation(replace_masks=True)
        self.masks_visible = True
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
        self._refresh_segmentation(replace_masks=True)
        self.masks_visible = True
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
            self.replace_segmentation(frame, masks)

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
            self.replace_segmentation(frame, np.zeros_like(frame.masks))
            if hasattr(self.stack, 'tracked_centroids'):
                t = self.stack.tracked_centroids
                self.stack.tracked_centroids = t[t.frame != frame.frame_number]
                self.left_toolbar.also_save_tracking.setChecked(True)

            if hasattr(frame, 'stored_mask_overlay'):
                del frame.stored_mask_overlay

    def _progress_bar(self, iterable, desc: str = None, length: int | None = None, leave: bool = True):
        if length is None:
            length = len(iterable)

        if self.progress_widget is not None:
            self.statusBar().removeWidget(self.progress_widget)
            self.progress_widget = None

        if length <= 1:
            return iterable
        else:
            # Initialize tqdm progress bar
            tqdm_bar = tqdm(iterable, desc=desc, total=length, leave=leave)

            # Initialize QProgressBar
            qprogress_bar = QProgressBar()
            qprogress_bar.setMaximum(length)

            # Set size policy to match the status bar width
            qprogress_bar.setFixedHeight(int(self.statusBar().height() * 0.8))
            qprogress_bar.setFixedWidth(int(self.statusBar().width() * 0.2))

            # Temporarily hide existing permanent status bar widgets
            self.status_coordinates.setVisible(False)
            self.status_pixel_value.setVisible(False)

            self.progress_widget = qprogress_bar
            self.statusBar().addPermanentWidget(qprogress_bar)

            # Custom iterator to update both progress bars
            def custom_iterator():
                self.is_iterating = True
                self.cancel_iter = False
                try:
                    for i, item in enumerate(iterable):
                        QApplication.processEvents()  # allow updates, check for key presses
                        if self.cancel_iter:
                            break
                        yield item
                        tqdm_bar.update(1)
                        qprogress_bar.setValue(i + 1)
                finally:
                    tqdm_bar.close()
                    self.statusBar().removeWidget(qprogress_bar)
                    self.progress_widget = None
                    # Restore existing permanent status bar widgets
                    self.status_coordinates.setVisible(True)
                    self.status_pixel_value.setVisible(True)
                    self.is_iterating = False
                    if self.cancel_iter:
                        QMessageBox.warning(self, 'Operation Cancelled', 'Operation cancelled by user.')

            return custom_iterator()

    def replace_segmentation(self, frame, masks):
        """
        Replace frame cells and outlines using frame.masks as reference.
        Called after any re-segmentation operation.
        """
        frame.masks = masks
        frame.outlines = masks_to_outlines(frame.masks)
        frame.n_cells = np.max(frame.masks)

        frame.cells = np.array(
            [
                Cell(n, outline=outline, parent=frame)
                for n, outline in enumerate(outlines_list(frame.masks))
            ]
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

        self.left_toolbar.red_threshold = self.frame.red_fluor_threshold
        self.left_toolbar.green_threshold = self.frame.green_fluor_threshold
        self.FUCCI_dropdown = 3
        self._FUCCI_overlay()

    def delete_particle_head(self):
        """
        Delete all masks preceding (and including) the current frame for the selected particle.
        """
        if not self.file_loaded:
            return
        if not hasattr(self.stack, 'tracked_centroids'):
            self.delete_cell(self.selected_cell)
            return

        else:
            particle_n = self.selected_particle_n
            if particle_n is None:
                return
            self._mock_delete(self.selected_cell_n)
            current_frame_n = self.frame_number

            to_delete = [cell for cell in self.stack.get_particle(particle_n) if cell.frame <= current_frame_n]

            if len(to_delete) == 0:
                return

            self.left_toolbar.also_save_tracking.setChecked(True)
            command = DeleteCellsCommand(
                self, cells=to_delete, description=f'Delete particle {particle_n} tail from frame {current_frame_n}'
            )

            self.undo_stack.push(command)

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
            self.delete_cell(self.selected_cell)
            return

        else:
            particle_n = self.selected_particle_n
            if particle_n is None:
                return
            self._mock_delete(self.selected_cell_n)

            current_frame_n = self.frame_number

            to_delete = [cell for cell in self.stack.get_particle(particle_n) if cell.frame >= current_frame_n]

            if len(to_delete) == 0:
                return

            self.left_toolbar.also_save_tracking.setChecked(True)
            command = DeleteCellsCommand(
                self, cells=to_delete, description=f'Delete particle {particle_n} tail from frame {current_frame_n}'
            )
            self.undo_stack.push(command)

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
            self.delete_cell(self.selected_cell_n)
            return

        else:
            if particle_n is None:
                particle_n = self.selected_particle_n
            if particle_n is None:
                return

            self._mock_delete(self.cell_from_particle(particle_n))

            t = self.stack.tracked_centroids
            self.left_toolbar.also_save_tracking.setChecked(True)


            cell_numbers, frame_numbers = np.array(t[t.particle == particle_n][['cell_number', 'frame']]).T
            to_delete = self.stack.get_particle(particle_n)

            command = DeleteCellsCommand(self, cells=to_delete, description=f'Delete particle {particle_n}')
            self.undo_stack.push(command)
            self._update_tracking_overlay()

    def _mock_delete(self, cell_n):
        """
        Mock delete a cell by hiding it from the display.
        """
        cell=self.frame.cells[cell_n]
        # faux deselect the cell
        self.canvas.add_cell_highlight(cell_n, color='none', layer='selection')

        # remove cell mask
        self.canvas.redraw_cell_mask(cell, color='none')

        # remove cell outline
        self.frame.outlines[cell.mask] = False
        self.canvas.update_outlines()

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

        self.canvas.draw_masks()
        self.canvas.draw_masks_parallel()

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
        # enforce checkbox exclusivity
        if sender == self.left_toolbar.highlight_track_ends_button and sender.isChecked():
            self.left_toolbar.highlight_mitoses_button.setChecked(False)
        elif sender == self.left_toolbar.highlight_mitoses_button and sender.isChecked():
            self.left_toolbar.highlight_track_ends_button.setChecked(False)

        if not self.file_loaded:
            return

        if hasattr(self.stack, 'tracked_centroids'):
            self.left_toolbar.update_tracking_label(self.stack.tracked_centroids['particle'].nunique())
        if hasattr(self.stack, 'mitoses'):
            self.left_toolbar.update_mitoses_label(len(self.stack.mitoses))

        if self.left_toolbar.tabbed_widget.currentIndex() != 2:
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
        self.left_toolbar.coverslip_height.setText(f'{self.frame.coverslip_height:.2f}')

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
                        coverslip_height = None
                    else:
                        coverslip_height = float(coverslip_height)

                    coverslip_prominence = self.left_toolbar.coverslip_prominence.text()
                    if coverslip_prominence == '':
                        coverslip_prominence = 0.01
                    else:
                        coverslip_prominence = float(coverslip_prominence)

                    self.measure_heights(frame, peak_prominence, coverslip_prominence, coverslip_height)
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
        
        peak_prominence = self.left_toolbar.coverslip_prominence.text()
        if peak_prominence == '':
            peak_prominence = 0.01
        else:
            peak_prominence = float(peak_prominence)

        for frame in self._progress_bar(frames):
            frame.coverslip_height = self.calibrate_coverslip_height(frame, prominence=peak_prominence)

        self.left_toolbar.coverslip_height.setText(f'{frame.coverslip_height:.2f}')

    def calibrate_coverslip_height(self, frames, membrane_channel=2, prominence=None):
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

        if isinstance(frames, SegmentedImage):
            frames = [frames]

        z_profile = []
        for z_index in range(frames[0].zstack.shape[0]):
            if self.is_grayscale:
                z_profile.append(np.mean(np.concatenate([frame.zstack[z_index].flatten() for frame in frames])))
            else:
                z_profile.append(
                    np.mean(np.concatenate([frame.zstack[z_index, ..., membrane_channel].flatten() for frame in frames]))
                )

        if not hasattr(self.frame, 'z_scale'):
            print(f'No z-scale available for {self.frame.name}. Defaulting to 1.')
            self.left_toolbar.z_size = 1.0
        scale = self.frame.z_scale

        coverslip_height = get_coverslip_z(z_profile, scale=scale, precision=0.01, prominence=prominence)
        return coverslip_height

    def _measure_heights(self):
        if not self.file_loaded:
            return
        if self.left_toolbar.volumes_on_stack.isChecked():
            frames = self.stack.frames
        else:
            frames = self.frame

        peak_prominence = self.left_toolbar.peak_prominence.text()
        if peak_prominence == '':
            peak_prominence = 0.01
        else:
            peak_prominence = float(peak_prominence)

        coverslip_height = self.left_toolbar.coverslip_height.text()
        if coverslip_height == '':
            coverslip_height = None
        else:
            coverslip_height = float(coverslip_height)

        self.measure_heights(frames, peak_prominence, coverslip_height)
        self._show_seg_overlay()
        self.left_toolbar.volume_button.setEnabled(True)
        self._export_heights_action.setEnabled(True)
        self.left_toolbar.coverslip_height.setText(f'{self.frame.coverslip_height:.2f}')

    def measure_heights(self, frames, peak_prominence=0.01, coverslip_prominence=0.01, coverslip_height=None, membrane_channel=2):
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

        if isinstance(frames, SegmentedImage):
            frames = [frames]

        for frame in self._progress_bar(frames):
            if not hasattr(frame, 'zstack'):
                raise ValueError(f'No z-stack available to measure heights for {frame.name}.')
            else:
                if not coverslip_height:
                    coverslip_height = self.calibrate_coverslip_height(frame, prominence=coverslip_prominence)
                frame.coverslip_height = coverslip_height
                if self.is_grayscale:
                    membrane = frame.zstack
                else:
                    membrane = frame.zstack[..., membrane_channel]  # TODO: allow user to specify membrane channel
                frame.heights = get_heights(membrane, peak_prominence=peak_prominence)
                frame.to_heightmap()

    def _compute_spherical_volumes(self):
        if not self.file_loaded:
            return

        if self.left_toolbar.volumes_on_stack.isChecked():
            frames = self.stack.frames
        else:
            frames = [self.frame]

        if self.left_toolbar.xy_size is None:
            print('No xy scale specified. Defaulting to 0.1625.')
            self.left_toolbar.xy_size = 0.1625

        self.get_spherical_volumes(frames, scale=self.left_toolbar.xy_size)

        self._refresh_right_toolbar('volume')

    def get_spherical_volumes(self, frames, scale=None):
        """
        Compute the volumes of cells if their areas are treated as circular cross-sections of spheres.
        This is used for estimating the volumes of cells in suspension.

        Parameters
        ----------
        frames : list of Frame
            The frames to compute spherical volumes for
        """
        if scale is None:
            print('No xy scale specified. Defaulting to 0.1625.')
            scale=0.1625

        for frame in self._progress_bar(frames):
            frame.scale=scale
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
            self.left_toolbar.get_coverslip_height_button.setEnabled(True)
            self.left_toolbar.get_heights_button.setEnabled(True)
            self.left_toolbar.peak_prominence.setEnabled(True)
        else:
            self.frame.img = self.frame.img
            self.zstack_slider.setVisible(False)
            self.is_zstack = False
            self.left_toolbar.get_coverslip_height_button.setEnabled(False)
            self.left_toolbar.get_heights_button.setEnabled(False)
            self.left_toolbar.peak_prominence.setEnabled(False)

        if self.is_zstack or hasattr(self.frame, 'heights'):
            self.left_toolbar.volume_button.setEnabled(True)

            if hasattr(self.frame, 'coverslip_height'):
                self.left_toolbar.coverslip_height.setText(f'{self.frame.coverslip_height:.2f}')
            else:
                self.left_toolbar.coverslip_height.setText('')
        else:
            self.left_toolbar.volume_button.setEnabled(False)
            self.left_toolbar.coverslip_height.setText('')

        self._export_heights_action.setEnabled(hasattr(self.frame, 'heights'))

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
            status_text='None'
            prompt_text=''
        else:
            status_text=str(cell_n)
            prompt_text=str(cell_n)

        self.selected_cell_prompt.blockSignals(True)
        self.status_cell.setText(f'Selected Cell: {status_text}')
        self.selected_cell_prompt.setText(prompt_text)
        self.selected_cell_prompt.blockSignals(False)

    def _update_tracking_ID_label(self, tracking_ID):
        """Update the status bar with the current tracking ID."""
        if tracking_ID is None:
            prompt_text=''
        else:
            prompt_text=str(tracking_ID)

        self.selected_particle_prompt.blockSignals(True)
        self.selected_particle_prompt.setText(prompt_text)
        self.selected_particle_prompt.blockSignals(False)

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
        self.stack.track_centroids(**kwargs)

    def _recolor_tracks(self, draw: bool = True):
        # recolor cells so each particle has one color over time
        for frame in self.stack.frames:
            if hasattr(frame, 'stored_mask_overlay'):
                del frame.stored_mask_overlay

        t = self.stack.tracked_centroids
        colors = self.canvas.random_color_ID(t['particle'].max() + 1)
        t['color'] = colors[t['particle']]
        for frame in self.stack.frames:
            tracked_frame = t[t.frame == frame.frame_number].sort_values('cell_number')
            frame.set_cell_attrs('color_ID', tracked_frame['color'])

        if draw:
            self.canvas.draw_masks_parallel()

    def particle_from_cell(self, cell_number: int, frame_number: int = None):
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

    def cell_from_particle(self, particle: int, frame_number: int = None):
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
        current_particle_n = self.selected_particle_n

        # if the selected particle has no head, nothing to split
        first_frame_cell = self.stack.get_particle(current_particle_n)[0]
        if self.frame_number == first_frame_cell.frame:
            return

        # Apply visual changes to the current frame
        current_cell_n = self.cell_from_particle(current_particle_n)
        current_cell_color = first_frame_cell.color_ID
        self.canvas.clear_overlay('selection')
        new_color = self.canvas.random_color_ID(ignore=current_cell_color)
        self._mock_select_recolor(self.frame.cells[current_cell_n], color=new_color)

        # Create and execute the split command
        command = SplitParticleTracksCommand(self, current_particle_n, self.frame_number, color=new_color)
        self.undo_stack.push(command)

        self.select_cell(cell=current_cell_n)
        self.left_toolbar.also_save_tracking.setChecked(True)
        self._plot_particle_statistic()

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

            if (
                self.stack.tracked_centroids[self.stack.tracked_centroids.particle == first_particle]['frame'].min()
                >= self.frame_number
            ):
                return  # first particle doesn't have a head, nothing to merge
            
            # Apply visual changes to the current frame
            cell2 = self.frame.cells[self.cell_from_particle(second_particle)]
            merge_color = self.stack.get_particle(first_particle)[0].color_ID
            self.canvas.clear_overlay('selection')
            self._mock_select_recolor(cell2, merge_color)

            # Create and execute the merge command
            command = MergeParticleTracksCommand(self, first_particle, second_particle, self.frame_number)
            self.undo_stack.push(command)

            self.left_toolbar.also_save_tracking.setChecked(True)
            self.select_cell(particle=first_particle)

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

    def _refresh_segmentation(self, replace_masks=False):
        """Redraw the outlines and masks overlays."""
        if not self.file_loaded:
            return
        if replace_masks and hasattr(self.frame, 'stored_mask_overlay'):
            del self.frame.stored_mask_overlay

        self.canvas.update_outlines()
        self.canvas.overlay_masks()
        self._update_tracking_overlay()
        self._update_ROIs_label()

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
        else:
            img = frame.img.reshape(1, *frame.img.shape)

        if img.ndim == 3:
            img = img.reshape(*img.shape, 1)

        bounds = []
        for z_slice in img:
            bounds.append(get_quantile(z_slice, q=(1, 99), mask_zeros=True))

        if len(bounds) == 1:
            bounds = bounds[0]
        return bounds

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
            script_globals=self.globals_dict.copy()
            script_globals['progress'] = self._progress_bar
            self.script_window = ScriptWindow(self, script_globals, self.locals_dict)
            self.script_window.show()

    def _show_undo_history(self):
        if self.undo_window is None or not self.undo_window.isVisible():
            self.undo_window = UndoHistoryWindow(self.undo_stack, parent=self)
        else:
            self.undo_window.raise_()
            self.undo_window.activateWindow()

    def select_cell(self, particle=None, cell=None):
        """
        Select a cell or particle by number. If None, selection is cleared.

        Parameters
        ----------
        particle : int
            The particle number to select.
        cell : int
            The cell number to select. If both particle and cell are specified, particle takes precedence.
        """
        if self.FUCCI_mode:  # classifying FUCCI, no cell selection
            return

        if particle is not None:  # select by particle number
            self.selected_particle_n = particle
            self.selected_cell_n = self.cell_from_particle(particle)
        elif cell is not None:  # select by cell number
            self.selected_cell_n = cell
            self.selected_particle_n = self.particle_from_cell(cell)
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
                if current_cell_n >= 0: # clicked on a cell

                    # delete cell from all frames
                    if (event.modifiers() & Qt.KeyboardModifier.ControlModifier) and (
                        event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                    ):
                        particle = self.particle_from_cell(current_cell_n, self.frame_number)
                        if particle is not None:
                            self.delete_particle(particle_n=particle)

                    # delete cell from current frame
                    elif event.modifiers() == Qt.KeyboardModifier.ControlModifier:
                        self.delete_cell(current_cell_n)

                        if current_particle_n is None: # no tracking data
                            self.select_cell(None)  # just deselect the cell
                        else: # tracking data available
                            self.select_cell(particle=current_particle_n) # reselect the particle
                    
                    # merge cells in all frames
                    elif (event.modifiers() & Qt.KeyboardModifier.AltModifier) and (
                        event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                    ):
                        particle = self.particle_from_cell(current_cell_n, self.frame_number)
                        if particle is not None:
                            self.merge_particle_masks(self.selected_particle_n, particle)

                    # merge cells in current frame
                    elif event.modifiers() == Qt.KeyboardModifier.AltModifier and self.selected_cell_n is not None:
                        self.merge_cell_masks(self.selected_cell_n, current_cell_n)
                        self.select_cell(cell=self.selected_cell_n)  # reselect the merged cell

                    # merge particles at current frame
                    elif event.modifiers() == Qt.KeyboardModifier.ShiftModifier:
                        if self.selected_particle_n is not None:
                            second_particle = self.particle_from_cell(current_cell_n)
                            if second_particle is not None:  # if a particle is found
                                self.merge_particle_tracks(self.selected_particle_n, second_particle)
                        self.select_cell(cell=current_cell_n)

                    # deselect current cell
                    elif current_cell_n == self.selected_cell_n:
                        self.select_cell(None)

                    # select the cell
                    else:
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

    def _close_cell_roi(self):
        """Close the cell ROI and add the new cell mask to the frame."""
        self.drawing_cell_roi = False
        enclosed_pixels = self.cell_roi.enclosed_pixels()
        # remove pixels outside the image bounds
        enclosed_pixels = enclosed_pixels[
            (enclosed_pixels[:, 0] >= 0)
            & (enclosed_pixels[:, 0] < self.frame.masks.shape[0])
            & (enclosed_pixels[:, 1] >= 0)
            & (enclosed_pixels[:, 1] < self.frame.masks.shape[1])
        ]
        self.add_cell(enclosed_pixels)
        self.cell_roi.clearPoints()

    def pixels_to_cell(self, pixels: np.ndarray, new_mask_n: int | None = None, frame: int | None = None):
        if frame is None:
            frame = self.frame
        if new_mask_n is None:
            new_mask_n = frame.n_cells

        binary_mask = np.zeros_like(frame.masks, dtype=bool)
        binary_mask[pixels[:, 0], pixels[:, 1]] = True
        binary_mask = binary_mask & (frame.masks == 0)

        if binary_mask.sum() < 5:  # check if the mask is more than 4 pixels (minimum for cellpose to generate an outline)
            return False, None

        outline = outlines_list(binary_mask)[0]
        cell = Cell(new_mask_n, outline=outline, parent=frame, red=False, green=False)

        return cell, binary_mask

    def random_color(self, n_samples: int | None = None):
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
    def add_cell(self, enclosed_pixels: np.ndarray, new_mask_n: int | None = None, frame: int | None = None):
        if frame is None:
            frame = self.frame
        if new_mask_n is None:
            new_mask_n = frame.n_cells

        selected_particle_n = self.selected_particle_n
        selected_cell_n = self.selected_cell_n
        cell, binary_mask = self.pixels_to_cell(enclosed_pixels, new_mask_n, frame)
        if cell is False:
            return False
        mask_command = AddCellCommand(
            self, cell=cell, mask=binary_mask, description=f'Add mask {new_mask_n} in frame {frame.frame_number}', refresh=True
        )
        self.undo_stack.push(mask_command)

        self.select_cell(particle=selected_particle_n, cell=selected_cell_n)
        return new_mask_n

    def delete_cell(self, cell_n: int, frame: SegmentedImage | None = None):
        if frame is None:
            frame = self.frame
        cell = frame.cells[cell_n]
        mask_command = DeleteCellCommand(
            self, cell=cell, description=f'Delete mask {cell_n} in frame {frame.frame_number}', refresh=True
        )
        self.undo_stack.push(mask_command)

    def _split_cell(self):
        self.drawing_cell_split = False
        self.split_cell_masks()
        self.cell_split.clearPoints()
        self.select_cell(particle=self.selected_particle_n)

    def split_cell_masks(self, min_size: int = 5):
        """
        Split cells that intersect with the current curve.

        Parameters
        ----------
        min_size : int
            The minimum size of a cell mask in pixels. Cells smaller than this will be merged with their largest neighbor.
        """
        curve_coords = np.array([(p.x(), p.y()) for p in self.cell_split.points]).astype(int)

        selected_particle_n = self.selected_particle_n
        selected_cell_n = self.selected_cell_n

        split_command=SplitCellCommand(self, curve_coords, min_size=min_size)

        if len(split_command.commands) > 0:
            self.undo_stack.push(split_command)
            self.select_cell(particle=selected_particle_n, cell=selected_cell_n)

    def merge_cell_masks(self, cell_n1: int, cell_n2: int, frame_number: int | None = None):
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
        if frame_number is None:
            frame_number = self.frame.frame_number

        if cell_n1 == cell_n2:
            return

        if frame_number is None:
            frame_number = self.frame_number
        # purge cell 2

        cell1 = self.frame.cells[cell_n1]
        cell2 = self.frame.cells[cell_n2]

        command = MergeCellsCommand(
            self, [cell1, cell2], description=f'Merge cells {cell_n1} and {cell_n2} in frame {frame_number}'
        )
        self.undo_stack.push(command)
        self.select_cell(cell=cell_n1)

        if hasattr(self.stack, 'tracked_centroids'):
            self.left_toolbar.also_save_tracking.setChecked(True)

        # self.check_cell_numbers() # for troubleshooting

    def merge_particle_masks(self, particle_n1: int, particle_n2: int):
        """
        Merges the masks of particle_n2 into particle_n1 in every frame.
        This is done by a series of cell mask merges and particle reassignments.

        Parameters
        ----------
        particle_n1 : int
            The particle number to merge into.
        particle_n2 : int
            The particle number to merge.
        """
        # same particle, no need to merge
        if particle_n1 == particle_n2:
            return
        # Apply visual change to current frame immediately
        cell2 = self.frame.cells[self.cell_from_particle(particle_n2)]
        self._mock_select_recolor(cell2, color=self.stack.get_particle(particle_n1)[0].color_ID)
        # Execute the merge
        command = MergeParticleMasksCommand(
            self, particle_n1, particle_n2, description=f'Merge particle {particle_n2} into particle {particle_n1}'
        )
        self.undo_stack.push(command)
        self.select_cell(particle=particle_n1)

    def _mock_select_recolor(self, cell, color=None):
        self.canvas.add_cell_highlight(
            cell.n, alpha=self.canvas.selected_cell_alpha, color=self.canvas.selected_cell_color
        )
        if color is not None:
            self.canvas.redraw_cell_mask(cell, color=color)

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

    def generate_outlines(self, frames: list[SegmentedImage]):
        """
        Generate cell outlines for a list of frames.

        Parameters
        ----------
        frames : list
            A list of Frame objects to generate outlines for.
        """

        for frame in self._progress_bar(frames, desc='Generating outlines'):
            outlines = outlines_list(frame.masks)
            for cell, outline in zip(frame.cells, outlines):
                cell.outline = outline
                cell.get_centroid()

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
            file_path, _ = QFileDialog.getSaveFileName(self, 'Save tracking data as...', filter='*.csv', directory=self.save_dir)

            if file_path == '':
                return
            self.save_dir = Path(file_path).parent

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
        file_path, _ = QFileDialog.getOpenFileName(self, 'Load tracking data...', filter='*.csv', directory=self.open_dir)
        if file_path == '':
            return

        self.open_dir = Path(file_path).parent

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
            folder_path = QFileDialog.getExistingDirectory(self, 'Save stack to folder...', directory=self.save_dir)
            if folder_path == '':
                return
            for frame in self._progress_bar(self.stack.frames):
                file_path = os.path.join(folder_path, os.path.basename(frame.name))
                self.save_frame(frame, file_path=file_path)
        else:
            file_path, _ = QFileDialog.getSaveFileName(self, 'Save frame as...', filter='*_seg.npy', directory=self.save_dir)
            folder_path = Path(file_path).parent
            if file_path == '':
                return
            if not file_path.endswith('_seg.npy'):
                file_path = file_path + '_seg.npy'
            self.save_frame(self.frame, file_path)

        print(f'Saved segmentation to {folder_path}')
        self.save_dir = folder_path
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
        try:  # fetch cell cycle data if available
            self._convert_red_green([frame])
            write_attrs = ['cell_cycles']

        except AttributeError:
            write_attrs = []

        if hasattr(frame, 'zstack'):
            frame.img = frame.zstack[self.zstack_number]

        frame.to_seg_npy(file_path, write_attrs=write_attrs, overwrite_img=True)
        frame.name = file_path

    def _export_ROIs(self):
        from roifile import roiwrite
        from segmentation_tools.utils import masks_to_rois

        save_output = FrameStackDialog.get_choice()
        if save_output is None:
            return

        elif save_output == 'stack':
            frames = self.stack.frames
        elif save_output == 'frame':
            frames = [self.frame]
        else:
            raise ValueError(f'Unexpected save_output value: {save_output}')

        roi_path, _ = QFileDialog.getSaveFileName(self, 'Save ROIs...', '', 'Zipped Archive (*.zip)')
        if roi_path == '':
            return

        for index, frame in enumerate(self._progress_bar(frames)):
            rois = masks_to_rois(frame.masks)
            if len(frames) > 1:
                frame_export_path = roi_path.replace('.zip', f'-{index}.zip')
            else:
                frame_export_path = roi_path
            for roi in rois:
                if roi is not None:
                    roiwrite(frame_export_path, roi)  # save each ROI to the correct path
            print(f'Saved {len(rois)} ROIs to {frame_export_path}')

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
        attrs = self._get_cell_frame_attrs(ignored={'n', 'frame', 'scale'})  # get all cell attributes except n and frame (redundant)

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
        if csv_df is None:
            csv_df = self._get_export()

        if columns == 'all':
            export = csv_df
        else:
            export = csv_df[columns]

        export.to_csv(file_path, index=False)

    def _export_heights(self):
        if not self.file_loaded:
            return

        save_output = FrameStackDialog.get_choice()
        if save_output is None:
            return

        elif save_output == 'stack':
            frames = self.stack.frames
        elif save_output == 'frame':
            frames = [self.frame]
        else:
            raise ValueError(f'Unexpected save_output value: {save_output}')
        save_path, _ = QFileDialog.getSaveFileName(
            self, 'Save heights...', filter='Numpy Archive (*.npz)', directory=self.save_dir
        )

        if save_path == '':
            return

        self.export_heights(frames, save_path)
        print(f'Saved heights to {save_path}')
        self.save_dir = Path(save_path).parent

    def export_heights(self, frames, file_path):
        heights = []
        coverslip_heights = []
        z_scales = []
        for frame in frames:
            try:
                heights.append(frame.heights)
            except AttributeError:
                QMessageBox.warning(self, 'Error', f'No heights found for frame {frame.frame_number}.')
                return
            coverslip_heights.append(getattr(frame, 'coverslip_height', 0))
            z_scales.append(getattr(frame, 'z_scale', 1))
        np.savez(file_path, heights=heights, coverslip_heights=coverslip_heights, z_scales=z_scales)

    def _import_heights(self):
        if not self.file_loaded:
            return
        # dialog to save either frame or stack
        heights_path, _ = QFileDialog.getOpenFileName(
            self, 'Load heights...', filter='Numpy Archive (*.npz)', directory=self.open_dir
        )
        if heights_path == '':
            return
        heights_file = np.load(heights_path)
        heights, coverslip_heights, z_scales = (
            heights_file['heights'],
            heights_file['coverslip_heights'],
            heights_file['z_scales'],
        )
        if len(heights) == 1:
            frames = [self.frame]
        elif len(heights) == len(self.stack.frames):
            frames = self.stack.frames
        else:
            raise ValueError(
                f'Number of heights arrays ({len(heights)}) does not match number of frames ({len(self.stack.frames)}).'
            )
            return

        for frame, height, coverslip_height, z_scale in zip(frames, heights, coverslip_heights, z_scales):
            frame.heights = height
            frame.coverslip_height = coverslip_height
            frame.z_scale = z_scale
        self.open_dir = Path(heights_path).parent

        print(f'Loaded heights from {heights_path} for {len(frames)} frames.')

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
                self.delete_cell(self.selected_cell_n)
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

    def _change_overlay_colormap(self):
        from .qt import ColormapPickerDialog

        dialog = ColormapPickerDialog(self.canvas.seg_stat_overlay.current_cmap, self)
        dialog.colormap_selected.connect(self.canvas.update_stat_overlay_lut)

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
        event.accept()
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        QTimer.singleShot(0, lambda: self.open_stack(natsorted(files)))

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

    def _save_screenshot(self):
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save screenshot as...', filter='*.png', directory=self.save_dir)
        if file_path == '':
            return
        self.save_screenshot(file_path=file_path)
        print(f'Saved screenshot to {file_path}')

    def _save_img_plot(self):
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save image plot as...', filter='*.png', directory=self.save_dir)
        if file_path == '':
            return
        self.canvas.save_img_plot(file_path)
        print(f'Saved image plot to {file_path}')

    def _save_seg_plot(self):
        file_path, _ = QFileDialog.getSaveFileName(self, 'Save segmentation plot as...', filter='*.png', directory=self.save_dir)
        if file_path == '':
            return
        self.canvas.save_seg_plot(file_path)
        print(f'Saved segmentation plot to {file_path}')

    def save_screenshot(self, file_path):
        """
        Take a screenshot of the main window and save it to a file.

        Parameters
        ----------
        file_path : str
            The file path to save the screenshot to.
        """
        screenshot = self.take_screenshot()
        if screenshot is None:
            return
        screenshot.save(file_path, 'png')  # Save to file
        self.save_dir = Path(file_path).parent

    def _export_window_video(self):
        self._export_video_prompt(screenshot='window')

    def _export_img_plot_video(self):
        self._export_video_prompt(screenshot='image')

    def _export_seg_plot_video(self):
        self._export_video_prompt(screenshot='segmentation')

    def _export_video_prompt(self, screenshot='window', fps=10):
        # Define supported formats with their extensions and descriptions
        formats = {
            'GIF (*.gif)': {
                'ext': 'gif',
                'ffmpeg_opts': {
                    'codec': None,  # ffmpeg will automatically use gif encoder
                    'extra': ['-filter_complex', '[0:v] split [a][b];[a] palettegen [p];[b][p] paletteuse'],
                },
            },
            'MP4 (*.mp4)': {
                'ext': 'mp4',
                'ffmpeg_opts': {
                    'codec': 'h264',  # Using built-in h264 encoder
                    'extra': ['-pix_fmt', 'yuv420p', '-crf', '23'],
                },
            },
            'MKV (FFV1) (*.mkv)': {'ext': 'mkv', 'ffmpeg_opts': {'codec': 'ffv1', 'extra': ['-level', '3']}},
            'Lossless PNG Video (*.mkv)': {'ext': 'mkv', 'ffmpeg_opts': {'codec': 'png', 'extra': []}},
        }

        filter_str = ';;'.join(formats.keys())
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, 'Save video as...', directory=self.save_dir, filter=filter_str
        )

        if file_path == '':
            return

        # Get the format settings based on selected filter
        format_settings = formats[selected_filter]

        # Ensure correct extension
        if not file_path.lower().endswith(f'.{format_settings["ext"]}'):
            file_path += f'.{format_settings["ext"]}'

        try:
            self.export_window_video(file_path, screenshot=screenshot, fps=fps, format_settings=format_settings['ffmpeg_opts'])
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save video: {str(e)}')
            return

        self.save_dir = Path(file_path).parent
        print(f'Saved video as {selected_filter} to {file_path}')

    def export_window_video(self, file_path: str, screenshot: str = 'window', fps: int = 10, format_settings: dict = None):
        """
        Save the stack with its current visual settings as a video.
        """
        import tempfile

        temp_dir = tempfile.mkdtemp()

        if screenshot == 'window':
            screenshot_func = self.take_screenshot
        elif screenshot == 'image':
            screenshot_func = self.canvas.save_img_plot
        elif screenshot == 'segmentation':
            screenshot_func = self.canvas.save_seg_plot
        else:
            raise ValueError(f'Invalid screenshot type: {screenshot}')

        # Ensure main window is active and process events
        self.activateWindow()
        self.raise_()

        # hide crosshairs while taking screenshots
        self.canvas.hide_crosshairs()

        # Process events and give a small delay for window activation
        QApplication.processEvents()
        QTimer.singleShot(100, lambda: None)

        # Convert each frame to PNG
        for frame_number in range(len(self.stack.frames)):
            self.change_current_frame(frame_number)
            QApplication.processEvents()
            QTimer.singleShot(0, lambda: None)

            frame_path = os.path.join(temp_dir, f'{frame_number:05d}.png')
            screenshot_func(frame_path)  # save a png of the current frame to the temp directory

        self.pngs_to_video(os.path.join(temp_dir, '%05d.png'), out_path=file_path, fps=fps, format_settings=format_settings)

        # Clean up temporary images
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
        self.canvas.show_crosshairs()

    def export_plot_video(self, format_settings, out_path=None, fps=10, plot='img'):
        pass

    @staticmethod
    def pngs_to_video(png_fmt, out_path, format_settings, fps=10):
        """
        Convert a series of PNG images to a video file using ffmpeg.

        Parameters
        ----------
        png_fmt : str
            The file path format string for the PNG images.
        video_path : str
            The file path to save the video to.
        fps : int
            The frames per second of the video.
        """
        import subprocess

        # Build ffmpeg command
        ffmpeg_cmd = ['ffmpeg', '-y', '-framerate', str(fps), '-i', png_fmt]

        if format_settings['codec']:
            ffmpeg_cmd.extend(['-c:v', format_settings['codec']])

        ffmpeg_cmd.extend(format_settings['extra'])
        ffmpeg_cmd.append(out_path)

        # Run ffmpeg
        subprocess.run(ffmpeg_cmd, check=True)

    @property
    def file_loaded(self):
        if not hasattr(self, '_file_loaded'):
            self._file_loaded = False
        return self._file_loaded

    @file_loaded.setter
    def file_loaded(self, value):
        self._file_loaded = value
        items = [
            self._save_action,
            self._save_as_action,
            self._export_csv_action,
            self._export_heights_action,
            self._export_ROIs_action,
            self._import_heights_action,
            self.import_images_action,
            self._save_img_plot_action,
            self._save_seg_plot_action,
            self._export_window_video_action,
            self._export_img_plot_video_action,
            self._export_seg_plot_video_action,
            self.clear_masks_action,
            self._generate_outlines_action,
            self._mend_gaps_action,
            self._remove_edge_masks_action,
            self._reorder_channels_action,
            self.rotate_clockwise_action,
            self.rotate_counterclockwise_action,
            self.delete_frame_action,
            self.make_substack_action,
        ]
        for item in items:
            item.setEnabled(value)

    def open_stack(self, files, image_shape=None):
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

        loaded_stack = self._load_files(files, image_shape=image_shape)
        if not loaded_stack:
            return

        self.stack = loaded_stack
        self.frame = self.stack.frames[0]

        self.globals_dict['stack'] = self.stack

        self.file_loaded = True
        if hasattr(self.stack, 'tracked_centroids'):
            self.left_toolbar.propagate_FUCCI_checkbox.setEnabled(True)
            self._recolor_tracks(draw=False)
        else:
            self.left_toolbar.propagate_FUCCI_checkbox.setChecked(False)
            self.left_toolbar.propagate_FUCCI_checkbox.setEnabled(False)

        if len(self.stack.frames) == 1:
            out_message = f'Loaded frame {self.stack.frames[0].name}.'
        else:
            out_message = f'Loaded stack {self.stack.name} with {len(self.stack.frames)} frames.'
        print(out_message)
        self.undo_stack.clear()
        self.statusBar().showMessage(out_message, 3000)
        self.open_dir = Path(self.stack.name).parent

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

        # reset visual settings
        self.canvas.img_plot.autoRange()

        self.left_toolbar.saved_visual_settings = [self.default_visual_settings for _ in range(4)]

        self._precompute_bounds()
        self.canvas.draw_masks_parallel()

    def _open_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, 'Open file(s)', filter='*seg.npy *.tif *.tiff *.nd2', directory=self.open_dir
        )
        if len(files) > 0:
            self.open_stack(files)

    def _load_files(self, files, image_shape=None):
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

            return stack

        elif len(img_files) > 0:  # image files
            from segmentation_viewer.io import read_image_file

            frames = []
            for file_path in img_files:
                file_path = Path(file_path)
                imgs = read_image_file(
                    str(file_path), image_shape=image_shape, progress_bar=self._progress_bar, desc=f'Loading {file_path.name}'
                )
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

    def make_substack(self, event=None, substack_frames=None):
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
            files, _ = QFileDialog.getOpenFileNames(
                self, 'Open image file(s)', filter='*.tif *.tiff *.nd2', directory=self.open_dir
            )
        elif isinstance(files, str):
            files = [files]

        if len(files) == 0:
            return

        self.open_dir = Path(files[0]).parent
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


class BaseCellMaskCommand(QUndoCommand):
    def __init__(
        self, main_window: MainWidget, cell: Cell, mask: np.ndarray, description: str = '', show: bool = True, parent=None
    ):
        super().__init__(description, parent)
        self.main_window = main_window
        self.cell = cell
        self.mask = mask

    def redo(self):
        self._redo_operation()

    def undo(self):
        self._undo_operation()

    def _redo_operation(self):
        raise NotImplementedError

    def _undo_operation(self):
        raise NotImplementedError

    def add_cell(self):
        frame = self.main_window.stack.frames[self.cell.frame]

        # ensure cell number is within bounds
        if self.cell.n > frame.n_cells:
            self.cell.n = frame.n_cells

        # add cell data
        frame.add_cell(self.cell, self.mask)

    def delete_cell(self):
        """
        Delete a cell object from the current frame.
        """
        frame = self.main_window.stack.frames[self.cell.frame]

        # remove cell data
        frame.remove_cell(self.cell, self.mask)


class AddCellMaskCommand(BaseCellMaskCommand):
    def __init__(
        self, main_window: MainWidget, cell: Cell, mask: np.ndarray, description: str = '', show: bool = True, parent=None
    ):
        description = description or f'Add cell {cell.n} to frame {cell.frame}'
        super().__init__(main_window, cell, mask, description, show, parent)

    def _redo_operation(self):
        self.add_cell()

    def _undo_operation(self):
        self.delete_cell()


class DeleteCellMaskCommand(BaseCellMaskCommand):
    def __init__(
        self,
        main_window: MainWidget,
        cell: Cell,
        mask: np.ndarray | None = None,
        description: str = '',
        show: bool = True,
        parent=None,
    ):
        if mask is None:
            mask = cell.mask
        description = description or f'Delete cell {cell.n} from frame {cell.frame}'
        super().__init__(main_window, cell, mask, description, show, parent)

    def _redo_operation(self):
        self.delete_cell()

    def _undo_operation(self):
        self.add_cell()


class BaseCellCommand(QUndoCommand):
    """
    Base class for cell add/delete operations.
    """

    def __init__(
        self,
        main_window: MainWidget,
        cell: Cell,
        mask: np.ndarray,
        description: str = '',
        show: bool = True,
        parent=None,
        refresh: bool = False,
    ):
        super().__init__(description, parent)
        self.main_window = main_window
        self.cell = cell
        self.mask = mask
        self.show = show
        self.cell_command = self._get_cell_command()
        self.color_command = self._get_color_command()
        self.tracking_command = self._get_tracking_command()
        self.commands = self._ordered_commands()
        self.refresh = refresh

    def _ordered_commands(self):
        raise NotImplementedError

    def _get_cell_command(self):
        raise NotImplementedError

    def _get_color_command(self):
        raise NotImplementedError

    def _get_tracking_command(self):
        """Create appropriate tracking command (to be implemented by subclasses)"""
        raise NotImplementedError

    def redo(self):
        if self.show:
            if self.main_window.frame_number != self.cell.frame:
                self.main_window.change_current_frame(self.cell.frame)
        for command in self.commands:
            command.redo()

        if self.refresh and self.main_window.frame_number == self.cell.frame:
            self.main_window._refresh_segmentation()
            self.main_window.select_cell(None)

    def undo(self):
        if self.show:
            if self.main_window.frame_number != self.cell.frame:
                self.main_window.change_current_frame(self.cell.frame)
        for command in reversed(self.commands):
            command.undo()

        if self.refresh and self.main_window.frame_number == self.cell.frame:
            self.main_window._refresh_segmentation()
            self.main_window.select_cell(None)


class AddCellCommand(BaseCellCommand):
    """
    Undoable command for adding a cell.
    """

    def __init__(
        self,
        main_window: MainWidget,
        cell: Cell,
        mask: np.ndarray,
        color=None,
        description: str = '',
        show: bool = True,
        parent=None,
        refresh: bool = False,
        row_args: dict = {},
    ):
        description = description or f'Add cell {cell.n} to frame {cell.frame}'
        if color is not None:
            self.new_cell_color = color
        elif hasattr(cell, 'color_ID'):
            self.new_cell_color = cell.color_ID
        else:
            self.new_cell_color = main_window.canvas.random_color_ID()
        self.row_args = row_args
        super().__init__(main_window, cell, mask, description, show, parent, refresh)

    def _ordered_commands(self):
        commands = [self.cell_command, self.color_command]
        if self.tracking_command is not None:
            commands.append(self.tracking_command)
        return commands

    def _get_cell_command(self):
        return AddCellMaskCommand(self.main_window, self.cell, self.mask, description='Add cell mask')

    def _get_color_command(self):
        return ChangeCellColorCommand(self.main_window.canvas, self.cell, self.new_cell_color, description='New cell color')

    def _get_tracking_command(self):
        if hasattr(self.main_window.stack, 'tracked_centroids'):
            return AddTrackingRowCommand(
                main_window=self.main_window,
                frame_number=self.cell.frame,
                cell_number=self.cell.n,
                cell=self.cell,
                row=None,
                row_args=self.row_args,
                description='Create tracking data for new cell',
            )
        else:
            return None


class DeleteCellCommand(BaseCellCommand):
    """
    Undoable command for deleting a cell.
    """

    def __init__(
        self,
        main_window: MainWidget,
        cell: Cell,
        mask: np.ndarray | None = None,
        description: str = '',
        show: bool = True,
        parent=None,
        refresh: bool = False,
    ):
        if mask is None:
            mask = cell.mask
        description = description or f'Delete cell {cell.n} from frame {cell.frame}'
        super().__init__(main_window, cell, mask, description, show, parent, refresh)

    def _ordered_commands(self):
        commands = [self.color_command, self.cell_command]
        if self.tracking_command is not None:
            commands.append(self.tracking_command)
        return commands

    def _get_cell_command(self):
        return DeleteCellMaskCommand(self.main_window, self.cell, self.mask, description='Delete cell mask')

    def _get_color_command(self):
        return ChangeCellColorCommand(self.main_window.canvas, self.cell, 'none', description='Remove cell color')

    def _get_tracking_command(self):
        if hasattr(self.main_window.stack, 'tracked_centroids'):
            tracking_data = self.main_window.stack.tracked_centroids
            row = tracking_data[(tracking_data.frame == self.cell.frame) & (tracking_data.cell_number == self.cell.n)]
            return DeleteTrackingRowCommand(
                main_window=self.main_window,
                frame_number=self.cell.frame,
                cell_number=self.cell.n,
                row=row,
                description='Remove tracking data for deleted cell',
            )
        else:
            return None


class DeleteCellsCommand(QUndoCommand):
    """
    Undoable command for adding or deleting a group of cells.
    """

    def __init__(self, main_window: MainWidget, cells: list[Cell], masks: list[np.ndarray] | None = None, description: str = ''):
        super().__init__(description)
        self.main_window = main_window
        self.cells = cells
        if masks is None:
            self.masks = [cell.mask for cell in cells]
        else:
            self.masks = masks
        self.cell_commands = []
        for cell, mask in zip(self.cells, self.masks):  # initialize cell commands
            command = DeleteCellCommand(self.main_window, cell, mask, description='Delete cell', show=False, refresh=True)
            self.cell_commands.append(command)

    def redo(self):
        self.main_window.select_cell(None)
        for command in self.main_window._progress_bar(self.cell_commands, leave=False):
            command.redo()

    def undo(self):
        self.main_window.select_cell(None)
        for command in self.main_window._progress_bar(self.cell_commands[::-1], leave=False):
            command.undo()


class ChangeCellColorCommand(QUndoCommand):
    """
    Undoable command for changing a cell's color.
    """

    def __init__(self, canvas: PyQtGraphCanvas, cell: Cell, color_ID: int, description: str = ''):
        super().__init__(description)
        self.canvas = canvas
        self.cell = cell
        self.color_ID = color_ID
        self.original_color_ID = getattr(cell, 'color_ID', 'none')

    def redo(self):
        self.cell.color_ID = self.color_ID
        self.canvas.redraw_cell_mask(self.cell)

    def undo(self):
        self.cell.color_ID = self.original_color_ID
        self.canvas.redraw_cell_mask(self.cell)


class MergeCellsCommand(QUndoCommand):
    """
    Undoable command for merging two cells.
    """

    def __init__(
        self,
        main_window: MainWidget,
        cells: list[Cell, Cell],
        masks: list[np.ndarray, np.ndarray] | None = None,
        description: str = '',
        show: bool = True,
    ):
        super().__init__(description)
        self.main_window = main_window
        frame = self.main_window.stack.frames[cells[0].frame]
        self.frame_number = frame.frame_number
        self.cell1, self.cell2 = cells
        if self.cell1.frame != self.cell2.frame:
            raise ValueError('Cells must be in the same frame to merge.')
        if masks is None:
            self.mask1, self.mask2 = self.cell1.mask, self.cell2.mask
        else:
            self.mask1, self.mask2 = masks
        self.merged_mask = self.mask1 | self.mask2
        self.merged_cell = self.cell1.copy()
        if self.merged_cell.n == frame.n_cells - 1:
            self.merged_cell.n -= 1
        self.merged_cell.outline = outlines_list(self.merged_mask)[0]

        row_args = {}
        if hasattr(self.main_window.stack, 'tracked_centroids'):
            particle1 = self.main_window.particle_from_cell(self.cell1.n, self.cell1.frame)
            row_args['particle'] = particle1

        self.commands = [
            DeleteCellCommand(self.main_window, self.cell1, self.mask1, description='Delete cell 1', show=False),
            DeleteCellCommand(self.main_window, self.cell2, self.mask2, description='Delete cell 2', show=False),
            AddCellCommand(
                self.main_window, self.merged_cell, self.merged_mask, description='Add merged cell', show=show, row_args=row_args
            ),
        ]

        if self.cell1.n < self.cell2.n:  # reorder delete commands to preserve cell numbering
            self.commands = [self.commands[1], self.commands[0], self.commands[2]]

        self.has_executed = False

    def redo(self):
        selection=self.main_window.selected_cell_n
        if selection == self.main_window.frame.n_cells - 1:
            selection -= 1
        if self.has_executed:
            self.main_window.select_cell(None)
        for command in self.commands:
            command.redo()
        if self.main_window.frame_number == self.frame_number:
            self.main_window._refresh_segmentation()
            self.main_window.select_cell(cell=selection) # reselect merged cell
        self.has_executed = True

    def undo(self):
        self.main_window.select_cell(None)
        for command in reversed(self.commands):
            command.undo()
        if self.main_window.frame_number == self.frame_number:
            self.main_window._refresh_segmentation()

class SplitCellCommand(QUndoCommand):
    def __init__(self, main_window: MainWidget, curve:np.array, min_size: int=0, description: str = '', refresh: bool = True):
        self.main_window = main_window
        self.frame_number = self.main_window.frame_number
        self.curve_coords = curve
        self.min_size = min_size
        self.refresh = refresh
        self.commands = self._create_commands()
        self.has_executed = False
        if description:
            self.description = description
        super().__init__(self.description)
    
    def _create_commands(self):
        from segmentation_tools.shape_operations import coords_to_mask, split_cell
        next_label = np.max(self.main_window.frame.masks)

        # Create a binary mask of the curve
        curve_mask = np.zeros_like(self.main_window.frame.masks, dtype=bool)
        for i in range(len(self.curve_coords) - 1):
            rr, cc = draw.line(self.curve_coords[i][0], self.curve_coords[i][1], self.curve_coords[i + 1][0], self.curve_coords[i + 1][1])
            # remove out-of-bounds coordinates
            inbound_coords = (cc >= 0) & (cc < curve_mask.shape[0]) & (rr >= 0) & (rr < curve_mask.shape[1])
            rr, cc = rr[inbound_coords], cc[inbound_coords]
            curve_mask[cc, rr] = True

        # Find unique labels that intersect with the curve
        intersected_labels = np.unique(self.main_window.frame.masks[curve_mask])
        intersected_labels = intersected_labels[intersected_labels != 0]

        commands = []
        split_IDs = []
        n_new = 0
        split_command = QUndoCommand()
        for label in intersected_labels:
            cell = self.main_window.frame.cells[label - 1]
            outline = cell.outline
            new_masks = split_cell(outline, self.curve_coords, min_area_threshold=self.min_size)
            if len(new_masks) == 0:  # no split
                continue
            split_IDs.append(label)
            n_new += len(new_masks)
            inheritor_cell = cell.copy()
            inheritor_mask = coords_to_mask(new_masks[0], shape=self.main_window.frame.masks.shape)
            inheritor_cell.outline = outlines_list(inheritor_mask)[0]
            if hasattr(self.main_window.stack, 'tracked_centroids'):
                inheritor_particle = self.main_window.particle_from_cell(cell.n)
                row_args = {'particle': inheritor_particle}
            else:
                row_args = {}
            commands.append(
                DeleteCellCommand(
                    self.main_window,
                    cell,
                    description=f'Delete unsplit mask {label} in frame {self.main_window.frame_number}',
                    parent=split_command,
                    show=False
                )
            )
            commands.append(
                AddCellCommand(
                    self.main_window,
                    inheritor_cell,
                    inheritor_mask,
                    description=f'Inheritor cell {label} in frame {self.main_window.frame_number}',
                    row_args=row_args,
                    parent=split_command,
                    show=False
                )
            )
            for i, new_mask in enumerate(new_masks[1:]):
                label_id = next_label + i
                new_mask = coords_to_mask(new_mask, shape=self.main_window.frame.masks.shape)
                new_outline = outlines_list(new_mask)[0]
                color_ID = self.main_window.canvas.random_color_ID()
                new_cell = Cell(label_id, new_outline, parent=self.main_window.frame, color_ID=color_ID)
                commands.append(
                    AddCellCommand(
                        self.main_window,
                        new_cell,
                        new_mask,
                        description=f'New split mask {label_id} in frame {self.main_window.frame_number}',
                        parent=split_command,
                        show=True
                    )
                )
        self.description = f'Split masks {split_IDs} into {n_new} masks in frame {self.main_window.frame_number}'
        return commands

    def redo(self):
        if self.has_executed:
            self.main_window.select_cell(None)
        else:
            self.has_executed = True

        for command in self.commands:
            command.redo()
        if self.refresh and self.main_window.frame_number == self.frame_number:
            self.main_window._refresh_segmentation()

    def undo(self):
        self.main_window.select_cell(None)
        for command in reversed(self.commands):
            command.undo()
        if self.refresh and self.main_window.frame_number == self.frame_number:
            self.main_window._refresh_segmentation()

class MergeParticleMasksCommand(QUndoCommand):
    """
    QUndoCommand for merging two particles' masks across all frames.

    This combines both the cell masks and particle trajectories:
    - For frames where both particles exist, their masks are merged
    - All frames with either particle1 or particle2 will be assigned to particle1
    """

    def __init__(self, main_window, particle1, particle2, description=None):
        """
        Initialize the merge particle masks command.

        Args:
            main_window: The main window containing the dataframe and cell data
            particle1: ID of the first particle (target particle)
            particle2: ID of the second particle (will be merged into particle1)
            description: Optional command description
        """
        super().__init__(description or f'Merge particle {particle2} masks into {particle1}')

        self.main_window = main_window
        self.particle1 = particle1
        self.particle2 = particle2

        # Will hold all the sub-commands created during execution
        self.commands = []

        # Store whether we've executed once already
        self.has_executed = False

    def _create_commands(self):
        """
        Create all the necessary sub-commands for merging the particle masks.
        This is done the first time redo() is called.
        """
        df = self.main_window.stack.tracked_centroids

        particle1_frames = list(df[df['particle'] == self.particle1]['frame'].values)
        particle2_frames = list(df[df['particle'] == self.particle2]['frame'].values)

        particle1 = self.main_window.stack.get_particle(self.particle1)
        particle2 = self.main_window.stack.get_particle(self.particle2)

        merged_color = particle1[0].color_ID

        # frames where particle2 masks need to be merged into particle1 masks
        merge_frames = set(particle1_frames).intersection(particle2_frames)

        # frames where particle2 cell needs to be relabeled as particle1
        relabel_frames = set(particle2_frames) - merge_frames

        for frame in merge_frames:
            cell1 = particle1[particle1_frames.index(frame)]
            cell2 = particle2[particle2_frames.index(frame)]
            mask1 = cell1.mask
            mask2 = cell2.mask

            merge_cmd = MergeCellsCommand(self.main_window, [cell1, cell2], [mask1, mask2], show=False)
            self.commands.append(merge_cmd)

        for frame in relabel_frames:
            cell2 = particle2[particle2_frames.index(frame)]

            # Create a command to reassign the cell from particle2 to particle1
            reassign_cmd = ReassignParticleCommand(self.main_window, cell2, self.particle2, self.particle1, color=merged_color)
            self.commands.append(reassign_cmd)

    def redo(self):
        """Execute the merge operation by executing all sub-commands."""
        if self.has_executed:
            self.main_window.select_cell(None)
        else:
            self._create_commands()
            self.has_executed = True
        
        for command in self.main_window._progress_bar(self.commands, leave=False):
            command.redo()

    def undo(self):
        """Undo the merge operation by undoing all sub-commands in reverse order."""
        for command in self.main_window._progress_bar(self.commands[::-1], leave=False):
            command.undo()
        self.main_window.select_cell(None)


class ReassignParticleCommand(QUndoCommand):
    """
    QUndoCommand for reassigning a cell from one particle to another.
    Also handles updating the cell color to match the new particle.
    """

    def __init__(
        self,
        main_window: MainWidget,
        cell: Cell,
        old_particle_id: int,
        new_particle_id: int,
        color: int | None = None,
        description: str | None = None,
    ):
        """
        Initialize the reassign particle command.

        Args:
            main_window: The main window containing the dataframe and cell data
            cell_id: The cell ID to reassign
            old_particle_id: The original particle ID
            new_particle_id: The new particle ID to assign
            description: Optional command description
        """
        super().__init__(description or f'Reassign particle {old_particle_id} to {new_particle_id} in frame {cell.frame}')

        self.main_window = main_window
        self.cell = cell
        self.cell_id = self.main_window.stack.get_tracking_row(self.cell).index.values[0]
        self.old_particle_id = old_particle_id
        self.new_particle_id = new_particle_id
        self.new_color_ID = color

        # Color command will be created during first execution
        self.color_command = self._get_color_command()

    def _get_color_command(self):
        # Get the cell object
        if self.new_color_ID is None:  # Get the color from the new particle
            new_particle = self.main_window.stack.get_particle(self.new_particle_id)
            if new_particle and len(new_particle) > 0:
                self.new_color_ID = new_particle[0].color_ID
            else:  # If the particle being added to doesn't exist, generate a new color
                try:
                    old_color_ID = self.main_window.stack.get_particle(self.old_particle_id)[0].color_ID
                except IndexError:
                    old_color_ID = None
                self.new_color_ID = self.main_window.canvas.random_color_ID(ignore=old_color_ID)

        # Create and execute the color command
        color_command = ChangeCellColorCommand(
            self.main_window.canvas,
            self.cell,
            self.new_color_ID,
            description=f'Update color for cell in frame {self.main_window.stack.tracked_centroids.at[self.cell_id, "frame"]}',
        )
        return color_command

    def redo(self):
        """Reassign the cell to the new particle ID and update its color."""
        # Make sure the cell exists in the dataframe
        if self.cell_id in self.main_window.stack.tracked_centroids.index:
            # Reassign the particle ID
            self.main_window.stack.tracked_centroids.at[self.cell_id, 'particle'] = self.new_particle_id

            # Handle color change if this is the first execution
            if self.color_command is None:
                self.color_command = self._get_color_command()
            self.color_command.redo()

    def undo(self):
        """Restore the cell to its original particle ID and color."""
        # First undo the color change if a color command exists
        self.color_command.undo()

        # Restore the original particle ID
        self.main_window.stack.tracked_centroids.at[self.cell_id, 'particle'] = self.old_particle_id


class MergeParticleTracksCommand(QUndoCommand):
    """
    QUndoCommand for merging two particle tracks at a specified frame.

    - All frames of particle2 from merge_frame onward will be assigned to particle1
    - If particle1 continues beyond merge_frame, it gets a new ID
    - If particle2 exists before merge_frame, it gets a new ID
    """

    def __init__(self, main_window, particle1, particle2, merge_frame, description=None):
        """
        Initialize the merge command.

        Args:
            main_window: The main window containing the dataframe and cell data
            particle1: ID of the first particle
            particle2: ID of the second particle
            merge_frame: Frame number where the merge occurs
            description: Optional command description
        """
        super().__init__(description or f'Merge particle {particle2} into particle {particle1} at frame {merge_frame}')

        self.main_window = main_window
        self.particle1 = particle1
        self.particle2 = particle2
        self.merge_frame = merge_frame

        self.commands = self._create_commands()

        # Store whether we've executed once already
        self.has_executed = False

        # Track new particle IDs that might be created
        self.new_particle1_id = None
        self.new_particle2_id = None

    def _create_commands(self):
        """
        Create all the necessary sub-commands for merging the particle tracks.
        This is done the first time redo() is called.
        """

        particle1 = np.array(self.main_window.stack.get_particle(self.particle1))
        particle2 = np.array(self.main_window.stack.get_particle(self.particle2))

        frames1 = np.array([cell.frame for cell in particle1])
        frames2 = np.array([cell.frame for cell in particle2])

        original_color1 = particle1[0].color_ID
        original_color2 = particle2[0].color_ID

        # Check if particle1 persists beyond merge_frame
        particle1_after = particle1[frames1>=self.merge_frame]
        particle2_before = particle2[frames2<self.merge_frame]
        particle2_after = particle2[frames2>=self.merge_frame]

        if len(particle2_after) == 0:
            return []

        next_ID = self.main_window.stack.tracked_centroids['particle'].max() + 1

        commands=[]
        # Create a new ID for particle1 continuation if needed
        if len(particle1_after) > 0:
            self.new_particle1_id = next_ID
            self.new_particle1_color = self.main_window.canvas.random_color_ID(ignore=[original_color1, original_color2])
            next_ID += 1

            # Create commands to reassign particle1 cells after merge_frame
            for cell1 in particle1_after:
                reassign_cmd = ReassignParticleCommand(
                    self.main_window, cell1, self.particle1, self.new_particle1_id, color=self.new_particle1_color
                )
                commands.append(reassign_cmd)

        # Create a new ID for particle2 before merge if needed
        if len(particle2_before) > 0:
            self.new_particle2_id = next_ID
            self.new_particle2_color = self.main_window.canvas.random_color_ID(ignore=[original_color1, original_color2])
            # Create commands to reassign particle2 cells before merge_frame
            for cell2 in particle2_before:
                reassign_cmd = ReassignParticleCommand(
                    self.main_window, cell2, self.particle2, self.new_particle2_id, color=self.new_particle2_color
                )
                commands.append(reassign_cmd)

        # Create commands for merging particle2 cells after merge_frame
        for cell2 in particle2_after:
            reassign_cmd = ReassignParticleCommand(
                self.main_window,
                cell2,
                self.particle2,
                self.particle1,
                color=original_color1,  # Use particle1's color for consistency
            )
            commands.append(reassign_cmd)

        return commands

    def redo(self):
        """Execute the merge operation by executing all sub-commands."""
        if self.has_executed:
            self.main_window.select_cell(None)
        else:
            self.has_executed = True

        for command in self.main_window._progress_bar(self.commands, leave=False):
            command.redo()
        self.main_window._update_tracking_overlay()

    def undo(self):
        """Undo the merge operation by undoing all sub-commands in reverse order."""
        self.main_window.select_cell(None)
        for command in self.main_window._progress_bar(self.commands[::-1], leave=False):
            command.undo()
        self.main_window._update_tracking_overlay()


class SplitParticleTracksCommand(QUndoCommand):
    """
    QUndoCommand for splitting a particle track at a specified frame.

    Assigns a new particle ID to all frames >= split_frame.
    """

    def __init__(self, main_window, particle_id, split_frame, color: int=None, description: str=None):
        """
        Initialize the split command.

        Args:
            main_window: The main window containing the dataframe and cell data
            particle_id: ID of the particle to split
            split_frame: Frame number where the split occurs
            description: Optional command description
        """
        super().__init__(description or f'Split particle {particle_id} at frame {split_frame}')

        self.main_window = main_window
        self.particle_id = particle_id
        self.split_frame = split_frame

        # Will hold all the sub-commands created during execution
        self.commands = []

        # Store whether we've executed once already
        self.has_executed = False

        # The new particle ID to be created
        self.new_particle_id = None

        if color is None:
            current_color = self.main_window.stack.get_particle(self.particle_id)[0].color_ID
            self.new_color = self.main_window.canvas.random_color_ID(ignore=current_color)
        else:
            self.new_color = color

    def _create_commands(self):
        """
        Create all the necessary sub-commands for splitting the particle track.
        This is done the first time redo() is called.
        """
        df = self.main_window.stack.tracked_centroids

        # Calculate new particle ID
        self.new_particle_id = df['particle'].max() + 1

        # Find all cells to split
        split_mask = (df['particle'] == self.particle_id) & (df['frame'] >= self.split_frame)

        # Get all cells from the original particle
        original_particle = self.main_window.stack.get_particle(self.particle_id)

        # Create commands for each cell to be reassigned
        for idx in df[split_mask].index:
            frame = df.at[idx, 'frame']
            cell = None

            # Find the corresponding cell from the particle
            for c in original_particle:
                if c.frame == frame:
                    cell = c
                    break

            if cell:
                reassign_cmd = ReassignParticleCommand(
                    self.main_window,
                    cell,
                    self.particle_id,
                    self.new_particle_id,
                    color=self.new_color,  # Use a new color for the split particle
                )
                self.commands.append(reassign_cmd)

    def redo(self):
        """Execute the split operation by executing all sub-commands."""
        if self.has_executed:
            self.main_window.select_cell(None)
        else:
            self._create_commands()
            self.has_executed = True

        for command in self.main_window._progress_bar(self.commands, leave=False):
            command.redo()
        self.main_window._update_tracking_overlay()

    def undo(self):
        """Undo the split operation by undoing all sub-commands in reverse order."""
        self.main_window.select_cell(None)
        for command in self.main_window._progress_bar(self.commands[::-1], leave=False):
            command.undo()
        self.main_window._update_tracking_overlay()


class BaseTrackingRowCommand(QUndoCommand):
    """
    Base class for tracking row operations.
    """

    def __init__(
        self, main_window: MainWidget, frame_number: int, cell_number: int, row: pd.DataFrame | None = None, description: str = ''
    ):
        super().__init__(description)
        self.main_window = main_window
        self.frame_number = frame_number
        self.cell_number = cell_number
        self.row = row

    def _determine_row(self):
        """Determine the row to work with, implemented by subclasses"""
        raise NotImplementedError

    def redo(self):
        if self.row is None:  # first execution, determine row
            self.row = self._determine_row()
        self._redo_operation()

    def undo(self):
        self._undo_operation()

    def _redo_operation(self):
        """Implemented by subclasses"""
        raise NotImplementedError

    def _undo_operation(self):
        """Implemented by subclasses"""
        raise NotImplementedError

    def delete_row(self):
        t = self.main_window.stack.tracked_centroids
        t.drop(self.row.index, inplace=True)
        t.loc[(t.frame == self.frame_number) & (t.cell_number > self.cell_number), 'cell_number'] -= 1

    def add_row(self):
        t = self.main_window.stack.tracked_centroids
        t.loc[(t.frame == self.frame_number) & (t.cell_number >= self.cell_number), 'cell_number'] += 1
        self.main_window.stack.tracked_centroids = pd.concat([t, self.row]).sort_values(['frame', 'particle'])


class AddTrackingRowCommand(BaseTrackingRowCommand):
    """
    Undoable command for adding a row to the tracking data.
    """

    def __init__(
        self,
        main_window: MainWidget,
        frame_number: int,
        cell_number: int,
        cell: Cell | None = None,
        row: pd.DataFrame | None = None,
        row_args: dict = {},
        description: str = 'Add Tracking Row',
    ):
        if cell is None:
            try:
                cell = main_window.stack.frames[frame_number].cells[cell_number]
            except IndexError:
                raise ValueError(f'Cell {cell_number} not found in frame {frame_number}.')
        self.cell = cell
        self.row_args = row_args
        super().__init__(main_window, frame_number, cell_number, row, description)

    def _determine_row(self):
        row = self.new_tracking_row()
        for k, v in self.row_args.items():
            row[k] = v
        return row

    def _redo_operation(self):
        self.add_row()

    def _undo_operation(self):
        self.delete_row()

    def new_tracking_row(self):
        t = self.main_window.stack.tracked_centroids
        new_particle_ID = t['particle'].max() + 1
        data = {
            'cell_number': self.cell.n,
            'y': self.cell.centroid[0],
            'x': self.cell.centroid[1],
            'frame': self.frame_number,
            'particle': new_particle_ID,
        }
        if hasattr(self.cell, 'color_ID'):
            data['color_ID'] = self.cell.color_ID
        placeholder_particle = {idx: None for idx in t.columns}
        placeholder_particle.update({k: v for k, v in data.items() if k in t.columns})
        row = pd.DataFrame([placeholder_particle], index=[t.index.max() + 1])
        return row


class DeleteTrackingRowCommand(BaseTrackingRowCommand):
    """
    Undoable command for deleting a row from the tracking data.
    """

    def __init__(
        self,
        main_window: MainWidget,
        frame_number: int,
        cell_number: int,
        row: pd.DataFrame | None = None,
        description: str = 'Delete Tracking Row',
    ):
        super().__init__(main_window, frame_number, cell_number, row, description)

    def _determine_row(self):
        return self.main_window.stack.tracked_centroids.loc[
            (self.main_window.stack.tracked_centroids.frame == self.frame_number)
            & (self.main_window.stack.tracked_centroids.cell_number == self.cell_number)
        ].copy()

    def _redo_operation(self):
        self.delete_row()

    def _undo_operation(self):
        self.add_row()


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
