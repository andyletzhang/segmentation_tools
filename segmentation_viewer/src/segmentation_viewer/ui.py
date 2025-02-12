import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator, QIntValidator
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .qt import CustomComboBox, FineScrubQRangeSlider

spacer = (0, 10)  # default spacer size (width, height)


def create_vertical_spacer(
    spacing=None,
    hSizePolicy=QSizePolicy.Policy.Fixed,
    vSizePolicy=QSizePolicy.Policy.Fixed,
):
    if spacing is None:
        spacing = spacer
    elif np.isscalar(spacing):
        spacing = (np.array(spacer) * spacing).astype(int)
    return QSpacerItem(*spacing, hSizePolicy, vSizePolicy)


class LeftToolbar(QScrollArea):
    def __init__(self, parent: QMainWindow):
        super().__init__(parent=parent)
        self.main_window = parent
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred
        )

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(10)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.lut_widget = self._create_lut_widget()
        self.voxel_size_widget = self._create_voxel_size_widget()
        self.tabbed_menu = self._create_tabbed_menu()
        self.command_line_button = self._create_command_line_button()
        self.save_widget = self._create_save_widget()

        self.RGB_mode()
        # Add all components
        for widget in (
            self.lut_widget,
            self.voxel_size_widget,
            self.tabbed_menu,
            self.command_line_button,
            self.save_widget,
        ):
            main_layout.addWidget(widget)

        self.setWidget(main_widget)
        return self

    def _create_lut_widget(self):
        widget = QWidget(objectName="bordered")
        layout = QVBoxLayout(widget)
        layout.setSpacing(0)

        # RGB checkboxes
        self.RGB_checkbox_layout = QVBoxLayout()
        layout.addLayout(self.RGB_checkbox_layout)
        layout.addSpacerItem(create_vertical_spacer(0.5))

        # Invert checkbox
        self.inverted_checkbox = QCheckBox("Invert [I]", widget)
        layout.addWidget(self.inverted_checkbox)
        layout.addItem(create_vertical_spacer())

        # Normalize section
        layout.addWidget(QLabel("Normalize by:", widget))
        layout.addWidget(self._create_normalize_widget())

        # Sliders
        self.slider_layout = QVBoxLayout()
        layout.addLayout(self.slider_layout)

        # Segmentation overlay
        layout.addWidget(self._create_overlay_checkboxes())

        return widget

    def _create_normalize_widget(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.normalize_frame_button = QRadioButton("Frame", self.main_window)
        self.normalize_stack_button = QRadioButton("Stack", self.main_window)
        self.normalize_custom_button = QRadioButton("LUT", self.main_window)

        for button in (
            self.normalize_frame_button,
            self.normalize_stack_button,
            self.normalize_custom_button,
        ):
            layout.addWidget(button)

        self.normalize_type = "frame"
        return widget

    @property
    def digit_width(self):
        return self.main_window.digit_width

    @property
    def normalize_type(self):
        """Get the state of the normalize buttons. Returns the selected button as a string."""
        button_status = [
            self.normalize_frame_button.isChecked(),
            self.normalize_stack_button.isChecked(),
            self.normalize_custom_button.isChecked(),
        ]
        button_names = np.array(["frame", "stack", "lut"])
        return button_names[button_status][0]

    @normalize_type.setter
    def normalize_type(self, value):
        """Set the state of the normalize buttons based on the input string."""
        button_names = np.array(["frame", "stack", "lut"])
        button_status = [value == name for name in button_names]
        self.normalize_frame_button.setChecked(button_status[0])
        self.normalize_stack_button.setChecked(button_status[1])
        self.normalize_custom_button.setChecked(button_status[2])

    def _create_overlay_checkboxes(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)

        self.masks_checkbox = QCheckBox("Masks [X]", self.main_window)
        self.outlines_checkbox = QCheckBox("Outlines [Z]", self.main_window)

        layout.addWidget(self.masks_checkbox)
        layout.addWidget(self.outlines_checkbox)

        return widget

    def _create_voxel_size_widget(self):
        widget = QWidget(objectName="bordered")
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("Voxel Size (μm):", self.main_window))

        size_layout = QHBoxLayout()
        self.xy_size_input = self._create_size_input()
        self.z_size_input = self._create_size_input()

        for label, input_field in [
            ("XY:", self.xy_size_input),
            ("Z:", self.z_size_input),
        ]:
            size_layout.addWidget(QLabel(label, self.main_window))
            size_layout.addWidget(input_field)

        layout.addLayout(size_layout)
        return widget

    @property
    def xy_size(self):
        xy_size_text = self.xy_size_input.text()
        if xy_size_text == "":
            return None
        else:
            return float(xy_size_text)

    @xy_size.setter
    def xy_size(self, value):
        self.xy_size_input.setText(str(value))
        self.main_window._update_voxel_size()

    @property
    def z_size(self):
        z_size_text = self.z_size_input.text()
        if z_size_text == "":
            return None
        else:
            return float(z_size_text)

    @z_size.setter
    def z_size(self, value):
        self.z_size_input.setText(str(value))
        self.main_window._update_voxel_size()

    def _create_size_input(self):
        size_input = QLineEdit(self, placeholderText="None")
        size_input.setValidator(QDoubleValidator(bottom=0))
        return size_input

    def _create_tabbed_menu(self):
        self.tabbed_widget = QTabWidget()
        self.current_tab = 0
        tabs = [
            ("Segmentation", self.get_segmentation_tab()),
            ("FUCCI", self.get_FUCCI_tab()),
            ("Tracking", self.get_tracking_tab()),
            ("Volumes", self.get_volumes_tab()),
        ]

        for name, widget in tabs:
            self.tabbed_widget.addTab(widget, name)

        return self.tabbed_widget

    def grayscale_mode(self):
        """Hide RGB GUI elements when a grayscale image is loaded."""
        self.is_grayscale = True
        self.clear_LUT_sliders()
        self.clear_RGB_checkboxes()
        self.add_grayscale_sliders(self.slider_layout)
        self.segmentation_channels_widget.hide()
        self.membrane_channel.setCurrentIndex(0)
        self.nuclear_channel.setCurrentIndex(0)

    def RGB_mode(self):
        """Show RGB GUI elements when an RGB image is loaded."""
        self.is_grayscale = False
        self.clear_LUT_sliders()
        self.clear_RGB_checkboxes()
        self.add_RGB_checkboxes(self.RGB_checkbox_layout)
        self.add_RGB_sliders(self.slider_layout)
        self.segmentation_channels_widget.show()
        self.show_grayscale_checkbox.setChecked(False)
        self.main_window._show_grayscale_toggled(False)

    def add_RGB_sliders(self, layout):
        self.LUT_range_sliders = []
        self.LUT_range_labels = []

        for label in ["R ", "G ", "B "]:
            slider_layout, slider, range_labels = labeled_LUT_slider(
                label, parent=self.lut_widget, digit_width=self.digit_width
            )
            layout.addLayout(slider_layout)
            self.LUT_range_sliders.append(slider)
            self.LUT_range_labels.append(range_labels)

            slider.valueChanged.connect(self.LUT_slider_changed)

    def LUT_slider_changed(self, event):
        """Update the LUTs when the sliders are moved."""
        self.normalize_custom_button.setChecked(True)
        self.main_window._set_LUTs()

    def update_LUT_labels(self):
        """Update the labels next to the LUT sliders with the current values."""
        for slider, labels in zip(self.LUT_range_sliders, self.LUT_range_labels):
            labels[0].setText(str(slider.value()[0]))
            labels[1].setText(str(slider.value()[1]))

    def add_grayscale_sliders(self, layout):
        self.LUT_range_sliders = []
        self.LUT_range_labels = []

        slider_layout, slider, range_labels = labeled_LUT_slider(parent=self)
        layout.addLayout(slider_layout)
        self.LUT_range_sliders.append(slider)
        self.LUT_range_labels.append(range_labels)

        slider.valueChanged.connect(self.LUT_slider_changed)

    def update_display(self):
        self.main_window._update_display()

    def toggle_grayscale(self):
        self.show_grayscale_checkbox.toggle()

    def toggle_inverted(self):
        self.inverted_checkbox.toggle()

    def add_channel_layout(self, channel_layout):
        self.membrane_channel = QComboBox(self)
        self.membrane_channel_label = QLabel("Membrane Channel:", self)
        self.membrane_channel.addItems(["Gray", "Red", "Green", "Blue"])
        self.membrane_channel.setCurrentIndex(3)
        self.membrane_channel.setFixedWidth(70)
        self.nuclear_channel = QComboBox(self)
        self.nuclear_channel_label = QLabel("Nuclear Channel:", self)
        self.nuclear_channel.addItems(["None", "Red", "Green", "Blue", "FUCCI"])
        self.nuclear_channel.setFixedWidth(70)

        membrane_tab_layout = QHBoxLayout()
        membrane_tab_layout.setSpacing(10)
        membrane_tab_layout.addWidget(self.membrane_channel_label)
        membrane_tab_layout.addWidget(self.membrane_channel)
        nuclear_layout = QHBoxLayout()
        nuclear_layout.setSpacing(5)
        nuclear_layout.addWidget(self.nuclear_channel_label)
        nuclear_layout.addWidget(self.nuclear_channel)

        channel_layout.addLayout(membrane_tab_layout)
        channel_layout.addLayout(nuclear_layout)

        return channel_layout

    def get_segmentation_tab(self):
        segmentation_tab = QWidget()
        segmentation_layout = QVBoxLayout(segmentation_tab)
        segmentation_layout.setSpacing(5)
        segmentation_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        segment_frame_widget = QWidget(objectName="bordered")
        segment_frame_layout = QVBoxLayout(segment_frame_widget)
        self.cell_diameter = QLineEdit(self, placeholderText="Auto")
        self.cell_diameter.setValidator(
            QDoubleValidator(bottom=0)
        )  # non-negative floats only
        self.cell_diameter.setFixedWidth(60)
        self.cell_diameter_calibrate = QPushButton("Calibrate", self)
        self.cell_diameter_calibrate.setFixedWidth(70)
        self.cell_diameter_layout = QHBoxLayout()
        self.cell_diameter_layout.setSpacing(5)
        self.cell_diameter_layout.addWidget(QLabel("Cell Diameter:", self))
        self.cell_diameter_layout.addWidget(self.cell_diameter)
        self.cell_diameter_layout.addWidget(self.cell_diameter_calibrate)

        # channel selection
        self.segmentation_channels_widget = QWidget()
        self.segmentation_channels_widget.setContentsMargins(0, 0, 0, 0)
        self.segmentation_channels_layout = QVBoxLayout(
            self.segmentation_channels_widget
        )
        self.add_channel_layout(self.segmentation_channels_layout)

        segmentation_button_layout = QHBoxLayout()
        self.segment_frame_button = QPushButton("Segment Frame", self)
        self.segment_stack_button = QPushButton("Segment Stack", self)
        segmentation_button_layout.addWidget(self.segment_frame_button)
        segmentation_button_layout.addWidget(self.segment_stack_button)

        # segmentation utilities
        segmentation_utils_widget = QWidget(objectName="bordered")
        segmentation_utils_layout = QVBoxLayout(segmentation_utils_widget)
        operate_on_label = QLabel("Operate on:", self)
        operate_on_layout = QHBoxLayout()
        self.segment_on_frame = QRadioButton("Frame", self)
        self.segment_on_stack = QRadioButton("Stack", self)
        self.segment_on_frame.setChecked(True)
        mend_remove_layout = QHBoxLayout()
        self.mend_gaps_button = QPushButton("Mend Gaps", self)
        self.remove_edge_masks_button = QPushButton("Remove Edge Masks", self)
        mend_remove_layout.addWidget(self.mend_gaps_button)
        mend_remove_layout.addWidget(self.remove_edge_masks_button)
        self.ROIs_label = QLabel("0 ROIs", self)
        gap_size_layout = QHBoxLayout()
        gap_size_label = QLabel("Gap Size:", self)
        self.gap_size = QLineEdit(self, placeholderText="Auto")
        self.gap_size.setValidator(
            QIntValidator(bottom=0)
        )  # non-negative integers only
        gap_size_layout.addWidget(gap_size_label)
        gap_size_layout.addWidget(self.gap_size)
        generate_remove_layout = QHBoxLayout()
        generate_outlines_button = QPushButton("Generate Outlines", self)
        clear_masks_button = QPushButton("Clear Masks", self)
        generate_remove_layout.addWidget(generate_outlines_button)
        generate_remove_layout.addWidget(clear_masks_button)
        operate_on_layout.addWidget(self.segment_on_frame)
        operate_on_layout.addWidget(self.segment_on_stack)
        segmentation_button_layout.addWidget(self.mend_gaps_button)
        segmentation_button_layout.addWidget(self.remove_edge_masks_button)

        segment_frame_layout.addLayout(self.cell_diameter_layout)
        segment_frame_layout.addWidget(self.segmentation_channels_widget)
        segment_frame_layout.addSpacerItem(create_vertical_spacer())
        segment_frame_layout.addWidget(self.ROIs_label)
        segment_frame_layout.addLayout(segmentation_button_layout)

        segmentation_utils_layout.addWidget(operate_on_label)
        segmentation_utils_layout.addLayout(operate_on_layout)
        segmentation_utils_layout.addLayout(mend_remove_layout)
        segmentation_utils_layout.addLayout(gap_size_layout)
        segmentation_utils_layout.addLayout(generate_remove_layout)

        segmentation_layout.addWidget(segment_frame_widget)
        segmentation_layout.addWidget(segmentation_utils_widget)

        self.mend_gaps_button.clicked.connect(self.main_window._mend_gaps_pressed)
        self.remove_edge_masks_button.clicked.connect(
            self.main_window._remove_edge_masks_pressed
        )
        self.cell_diameter.textChanged.connect(self.main_window._update_cell_diameter)
        self.cell_diameter_calibrate.clicked.connect(
            self.main_window._calibrate_diameter_pressed
        )
        self.segment_frame_button.clicked.connect(
            self.main_window._segment_frame_pressed
        )
        self.segment_stack_button.clicked.connect(
            self.main_window._segment_stack_pressed
        )
        generate_outlines_button.clicked.connect(
            self.main_window._generate_outlines_pressed
        )
        clear_masks_button.clicked.connect(self.main_window.clear_masks)

        return segmentation_tab

    def get_FUCCI_tab(self):
        FUCCI_tab = QWidget()
        FUCCI_tab_layout = QVBoxLayout(FUCCI_tab)
        FUCCI_tab_layout.setSpacing(10)
        FUCCI_tab_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        get_intensities_button = QPushButton("Get Cell Red/Green Intensities", self)
        get_tracked_FUCCI_button = QPushButton("FUCCI from tracks", self)
        measure_FUCCI_widget = QWidget(objectName="bordered")
        measure_FUCCI_layout = QVBoxLayout(measure_FUCCI_widget)

        red_threshold_layout = QHBoxLayout()
        red_threshold_label = QLabel("Red Threshold:", self)
        self.red_threshold_input = QLineEdit(self, placeholderText="Auto")
        self.red_threshold_input.setFixedWidth(60)
        self.red_threshold_input.setValidator(
            QDoubleValidator(bottom=0)
        )  # non-negative floats only
        red_threshold_layout.addWidget(red_threshold_label)
        red_threshold_layout.addWidget(self.red_threshold_input)

        green_threshold_layout = QHBoxLayout()
        green_threshold_label = QLabel("Green Threshold:", self)
        self.green_threshold_input = QLineEdit(self, placeholderText="Auto")
        self.green_threshold_input.setFixedWidth(60)
        self.green_threshold_input.setValidator(
            QDoubleValidator(bottom=0)
        )  # non-negative floats only
        green_threshold_layout.addWidget(green_threshold_label)
        green_threshold_layout.addWidget(self.green_threshold_input)

        percent_threshold_layout = QHBoxLayout()
        percent_threshold_label = QLabel("Minimum N/C Ratio:", self)
        self.percent_threshold_input = QLineEdit(self, placeholderText="0.15")
        self.percent_threshold_input.setFixedWidth(60)
        self.percent_threshold_input.setValidator(
            QDoubleValidator(bottom=0)
        )  # non-negative floats only
        percent_threshold_layout.addWidget(percent_threshold_label)
        percent_threshold_layout.addWidget(self.percent_threshold_input)

        FUCCI_button_layout = QHBoxLayout()
        FUCCI_button_layout.setSpacing(5)
        FUCCI_frame_button = QPushButton("Measure Frame", self)
        FUCCI_stack_button = QPushButton("Measure Stack", self)
        FUCCI_button_layout.addWidget(FUCCI_frame_button)
        FUCCI_button_layout.addWidget(FUCCI_stack_button)

        annotate_FUCCI_widget = QWidget(objectName="bordered")
        annotate_FUCCI_layout = QVBoxLayout(annotate_FUCCI_widget)
        FUCCI_overlay_layout = QHBoxLayout()
        overlay_label = QLabel("FUCCI overlay: ", self)
        self.FUCCI_dropdown = QComboBox(self)
        self.FUCCI_dropdown.addItems(["None", "Green", "Red", "All"])
        FUCCI_overlay_layout.addWidget(overlay_label)
        FUCCI_overlay_layout.addWidget(self.FUCCI_dropdown)
        self.FUCCI_checkbox = QCheckBox("Show FUCCI Channel", self)
        # clear FUCCI, propagate FUCCI
        self.propagate_FUCCI_checkbox = QCheckBox("Propagate FUCCI", self)
        self.propagate_FUCCI_checkbox.setEnabled(False)
        clear_frame_button = QPushButton("Clear Frame", self)
        clear_stack_button = QPushButton("Clear Stack", self)
        clear_FUCCI_layout = QHBoxLayout()
        clear_FUCCI_layout.addWidget(clear_frame_button)
        clear_FUCCI_layout.addWidget(clear_stack_button)

        measure_FUCCI_layout.addLayout(red_threshold_layout)
        measure_FUCCI_layout.addLayout(green_threshold_layout)
        measure_FUCCI_layout.addLayout(percent_threshold_layout)
        measure_FUCCI_layout.addSpacerItem(create_vertical_spacer())
        measure_FUCCI_layout.addLayout(FUCCI_button_layout)

        annotate_FUCCI_layout.addLayout(FUCCI_overlay_layout)
        annotate_FUCCI_layout.addWidget(self.FUCCI_checkbox)
        annotate_FUCCI_layout.addSpacerItem(create_vertical_spacer())
        annotate_FUCCI_layout.addWidget(self.propagate_FUCCI_checkbox)
        annotate_FUCCI_layout.addLayout(clear_FUCCI_layout)

        FUCCI_tab_layout.addWidget(get_intensities_button)
        FUCCI_tab_layout.addWidget(get_tracked_FUCCI_button)
        FUCCI_tab_layout.addWidget(measure_FUCCI_widget)
        FUCCI_tab_layout.addWidget(annotate_FUCCI_widget)

        get_intensities_button.clicked.connect(
            self.main_window.cell_red_green_intensities
        )
        get_tracked_FUCCI_button.clicked.connect(self.main_window.get_tracked_FUCCI)
        self.FUCCI_dropdown.currentIndexChanged.connect(
            self.main_window._FUCCI_overlay_changed
        )
        self.FUCCI_checkbox.stateChanged.connect(self.main_window._update_display)
        FUCCI_frame_button.clicked.connect(self.main_window._measure_FUCCI_frame)
        FUCCI_stack_button.clicked.connect(self.main_window._measure_FUCCI_stack)
        self.propagate_FUCCI_checkbox.stateChanged.connect(
            self.main_window._propagate_FUCCI_toggled
        )
        clear_frame_button.clicked.connect(self.main_window._clear_FUCCI_frame_pressed)
        clear_stack_button.clicked.connect(self.main_window._clear_FUCCI_stack_pressed)

        return FUCCI_tab

    @property
    def red_threshold(self):
        red_threshold_text = self.red_threshold_input.text()
        if red_threshold_text == "":
            return None
        else:
            return float(red_threshold_text)

    @red_threshold.setter
    def red_threshold(self, value):
        if value is None:
            self.red_threshold_input.setText("")
        else:
            self.red_threshold_input.setText(f"{value:.2f}")

    @property
    def green_threshold(self):
        green_threshold_text = self.green_threshold_input.text()
        if green_threshold_text == "":
            return None
        else:
            return float(green_threshold_text)

    @green_threshold.setter
    def green_threshold(self, value):
        if value is None:
            self.green_threshold_input.setText("")
        else:
            self.green_threshold_input.setText(f"{value:.2f}")

    @property
    def percent_threshold(self):
        percent_threshold_text = self.percent_threshold_input.text()
        if percent_threshold_text == "":
            return 0.15
        else:
            return float(percent_threshold_text)

    @percent_threshold.setter
    def percent_threshold(self, value):
        if value is None:
            self.percent_threshold_input.setText("")
        else:
            self.percent_threshold_input.setText(f"{value:.2f}")

    def get_volumes_tab(self):
        self.volumes_tab = QWidget()
        volumes_layout = QVBoxLayout(self.volumes_tab)
        volumes_layout.setSpacing(10)
        volumes_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        operate_on_label = QLabel("Operate on:", self)
        operate_on_layout = QHBoxLayout()
        self.volumes_on_frame = QRadioButton("Frame", self)
        self.volumes_on_stack = QRadioButton("Stack", self)
        operate_on_layout.addWidget(self.volumes_on_frame)
        operate_on_layout.addWidget(self.volumes_on_stack)
        self.volumes_on_frame.setChecked(True)
        self.get_heights_layout = QHBoxLayout()
        self.get_heights_button = QPushButton("Measure Heights", self)
        self.volume_button = QPushButton("Measure Volumes", self)
        self.get_heights_layout.addWidget(self.get_heights_button)
        self.get_heights_layout.addWidget(self.volume_button)
        peak_prominence_label = QLabel("Peak Prominence (0 to 1):", self)
        self.peak_prominence = QLineEdit(self, text="0.01", placeholderText="0.01")
        self.peak_prominence.setValidator(
            QDoubleValidator(bottom=0)
        )  # non-negative floats only
        self.peak_prominence.setFixedWidth(60)
        self.peak_prominence_layout = QHBoxLayout()
        self.peak_prominence_layout.addWidget(peak_prominence_label)
        self.peak_prominence_layout.addWidget(self.peak_prominence)
        self.get_coverslip_height_layout = QHBoxLayout()
        coverslip_height_label = QLabel("Coverslip Height (μm):", self)
        self.coverslip_height = QLineEdit(self, placeholderText="Auto")
        self.coverslip_height.setValidator(
            QDoubleValidator(bottom=0)
        )  # non-negative floats only
        self.coverslip_height.setFixedWidth(60)
        self.get_coverslip_height = QPushButton("Calibrate", self)
        self.get_coverslip_height_layout.addWidget(coverslip_height_label)
        self.get_coverslip_height_layout.addWidget(self.coverslip_height)
        self.get_coverslip_height_layout.addWidget(self.get_coverslip_height)
        self.get_spherical_volumes = QPushButton("Compute Spherical Volumes", self)

        volumes_layout.addWidget(operate_on_label)
        volumes_layout.addLayout(operate_on_layout)
        volumes_layout.addLayout(self.get_heights_layout)
        volumes_layout.addLayout(self.peak_prominence_layout)
        volumes_layout.addLayout(self.get_coverslip_height_layout)
        volumes_layout.addWidget(self.get_spherical_volumes)

        self.volume_button.clicked.connect(self.main_window._measure_volumes_pressed)
        self.get_heights_button.clicked.connect(
            self.main_window._measure_heights_pressed
        )
        self.get_coverslip_height.clicked.connect(
            self.main_window._calibrate_coverslip_height_pressed
        )
        self.get_spherical_volumes.clicked.connect(
            self.main_window._compute_spherical_volumes_pressed
        )

        return self.volumes_tab

    def get_tracking_tab(self):
        from .qt import CollapsibleWidget, bordered

        tracking_tab = QWidget()
        tracking_tab_layout = QVBoxLayout(tracking_tab)
        tracking_tab_layout.setSpacing(5)
        tracking_tab_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.tracking_range_layout = QFormLayout()
        range_label = QLabel("Search Range:  ", self)
        memory_label = QLabel("Memory:  ", self)
        self.memory_range = QLineEdit(self, placeholderText="0")
        self.tracking_range = QLineEdit(self, placeholderText="Auto")
        self.tracking_range.setValidator(
            QDoubleValidator(bottom=0)
        )  # non-negative floats only
        self.memory_range.setValidator(
            QIntValidator(bottom=0)
        )  # non-negative integers only
        self.tracking_range_layout.addRow(range_label, self.tracking_range)
        self.tracking_range_layout.addRow(memory_label, self.memory_range)

        self.track_centroids_button = QPushButton("Track Centroids", self)

        io_menu = QHBoxLayout()
        self.save_tracking_button = QPushButton("Save Tracking", self)
        self.load_tracking_button = QPushButton("Load Tracking", self)
        io_menu.addWidget(self.save_tracking_button)
        io_menu.addWidget(self.load_tracking_button)

        highlight_tracking_layout = QHBoxLayout()
        self.highlight_track_ends_button = QCheckBox("Highlight Track Ends", self)
        self.highlight_mitoses_button = QCheckBox("Highlight Mitoses", self)
        highlight_tracking_layout.addWidget(self.highlight_track_ends_button)
        highlight_tracking_layout.addWidget(self.highlight_mitoses_button)

        split_particle_button = QPushButton("Split Particle", self)
        delete_particle_label = QLabel("Delete Particle:", self)
        delete_particle_layout = QHBoxLayout()
        delete_head = QPushButton("Head", self)
        delete_tail = QPushButton("Tail", self)
        delete_all = QPushButton("All", self)
        delete_particle_layout.addWidget(delete_head)
        delete_particle_layout.addWidget(delete_tail)
        delete_particle_layout.addWidget(delete_all)
        clear_tracking_button = QPushButton("Clear Tracking", self)

        get_mitoses_button = QPushButton("Get Mitoses", self)
        get_mitoses_button.clicked.connect(self.main_window._get_mitoses_pressed)

        self.mitosis_inputs = [
            QLineEdit(self, placeholderText=text)
            for text in ["1.5", "1", "1", "1", "1", "1", "1"]
        ]
        self.mitoses_config_menu = QFormLayout()
        for label, line in zip(
            [
                "Distance Threshold:",
                "Score Cutoff:",
                "Mother Circularity:",
                "Daughter Circularity:",
                "Centroid Asymmetry:",
                "Centroid Angle:",
                "CoM Displacement:",
            ],
            self.mitosis_inputs,
        ):
            self.mitoses_config_menu.addRow(QLabel(label), line)

        track_centroids_widget = CollapsibleWidget(
            header_text="Track Centroids", parent=self.tabbed_widget
        )
        tracking_border = bordered(track_centroids_widget)

        mitoses_widget = CollapsibleWidget(
            header_text="Mitoses", parent=self.tabbed_widget
        )
        mitoses_border = bordered(mitoses_widget)

        track_centroids_widget.core_layout.addWidget(self.track_centroids_button)
        track_centroids_widget.core_layout.addLayout(io_menu)
        track_centroids_widget.addLayout(self.tracking_range_layout)
        track_centroids_widget.addWidget(split_particle_button)
        track_centroids_widget.addWidget(delete_particle_label)
        track_centroids_widget.addLayout(delete_particle_layout)
        track_centroids_widget.addWidget(clear_tracking_button)
        track_centroids_widget.hide_content()

        mitoses_widget.core_layout.addWidget(get_mitoses_button)
        mitoses_widget.addLayout(self.mitoses_config_menu)
        mitoses_widget.hide_content()

        tracking_tab_layout.addLayout(highlight_tracking_layout)
        tracking_tab_layout.addWidget(tracking_border)
        tracking_tab_layout.addWidget(mitoses_border)

        self.track_centroids_button.clicked.connect(
            self.main_window._track_centroids_pressed
        )
        self.tracking_range.returnPressed.connect(
            self.main_window._track_centroids_pressed
        )
        split_particle_button.clicked.connect(self.main_window.split_particle_tracks)
        clear_tracking_button.clicked.connect(self.main_window.clear_tracking)
        self.save_tracking_button.clicked.connect(self.main_window.save_tracking)
        self.load_tracking_button.clicked.connect(
            self.main_window._load_tracking_pressed
        )
        self.highlight_track_ends_button.stateChanged.connect(
            self.main_window._update_tracking_overlay
        )
        self.highlight_mitoses_button.stateChanged.connect(
            self.main_window._update_tracking_overlay
        )
        delete_head.clicked.connect(self.main_window.delete_particle_head)
        delete_tail.clicked.connect(self.main_window.delete_particle_tail)
        delete_all.clicked.connect(self.main_window.delete_particle)

        return tracking_tab

    def _create_save_widget(self):
        widget = QWidget(objectName="bordered")
        layout = QGridLayout(widget)
        layout.setVerticalSpacing(5)

        self.save_button = QPushButton("Save", self)
        self.save_as_button = QPushButton("Save As", self)
        self.save_stack = QCheckBox("Save Stack", self)
        self.also_save_tracking = QCheckBox("Save Tracking", self)

        self.save_stack.setChecked(True)

        layout.addWidget(self.save_button, 0, 0)
        layout.addWidget(self.save_as_button, 0, 1)
        layout.addWidget(self.save_stack, 1, 0)
        layout.addWidget(self.also_save_tracking, 1, 1)

        return widget

    def _create_command_line_button(self):
        self.command_line_button = QPushButton("Open Command Line", self)
        return self.command_line_button

    def _connect_signals(self):
        # Normalize signals
        self.inverted_checkbox.stateChanged.connect(self.main_window._invert_toggled)
        self.normalize_frame_button.toggled.connect(
            self.main_window._update_normalize_frame
        )
        self.normalize_stack_button.toggled.connect(
            self.main_window._update_normalize_frame
        )
        self.normalize_custom_button.toggled.connect(
            self.main_window._update_normalize_frame
        )

        # Segmentation overlay signals
        self.masks_checkbox.stateChanged.connect(self.set_masks_visibility)
        self.outlines_checkbox.stateChanged.connect(self.set_outlines_visibility)

        # Command line signal
        self.command_line_button.clicked.connect(self.main_window._open_command_line)

        # Voxel size signals
        self.xy_size_input.editingFinished.connect(self.main_window._update_voxel_size)
        self.z_size_input.editingFinished.connect(self.main_window._update_voxel_size)

        # Save signals
        self.save_button.clicked.connect(self.main_window._save_segmentation)
        self.save_as_button.clicked.connect(self.main_window._save_as_segmentation)

        # Tab switch signal
        self.tabbed_widget.currentChanged.connect(self.tab_switched)

    def set_masks_visibility(self):
        self.main_window.masks_visible = self.masks_checkbox.isChecked()

    def set_outlines_visibility(self):
        self.main_window.outlines_visible = self.outlines_checkbox.isChecked()

    def tab_switched(self, index):
        # save visual settings for the previous tab
        self.saved_visual_settings[self.current_tab] = self._visual_settings

        # load visual settings for the new tab
        self.current_tab = index
        self._visual_settings = self.saved_visual_settings[index]
        self.main_window._tab_switched(index)

    def clear_LUT_sliders(self):
        try:
            clear_layout(self.slider_layout)
            for slider in self.LUT_range_sliders:
                slider.deleteLater()
            self.LUT_range_sliders.clear()
        except AttributeError:
            pass

    def clear_RGB_checkboxes(self):
        try:
            clear_layout(self.RGB_checkbox_layout)
            for checkbox in self.RGB_checkboxes:
                checkbox.deleteLater()
            self.RGB_checkboxes.clear()
        except AttributeError:
            pass

    def add_RGB_checkboxes(self, layout):
        color_channels_layout = QHBoxLayout()
        color_channels_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        color_channels_layout.setSpacing(25)
        self.RGB_checkboxes = [
            QCheckBox(s, parent=self.lut_widget) for s in ["R", "G", "B"]
        ]
        for checkbox in self.RGB_checkboxes:
            checkbox.setChecked(True)
            color_channels_layout.addWidget(checkbox)

        self.show_grayscale_checkbox = QCheckBox("Grayscale")
        self.show_grayscale_checkbox.setChecked(False)

        layout.addSpacerItem(create_vertical_spacer())
        layout.addLayout(color_channels_layout)
        layout.addSpacerItem(create_vertical_spacer())
        layout.addWidget(self.show_grayscale_checkbox)

        for checkbox in self.RGB_checkboxes:
            checkbox.stateChanged.connect(self.update_display)
        self.show_grayscale_checkbox.stateChanged.connect(
            self.main_window._show_grayscale_toggled
        )

    @property
    def RGB_visible(self):
        if self.is_grayscale:
            return None
        else:
            return [checkbox.isChecked() for checkbox in self.RGB_checkboxes]

    @RGB_visible.setter
    def RGB_visible(self, RGB):
        if isinstance(RGB, bool):
            RGB = [RGB] * 3
        elif len(RGB) != 3:
            raise ValueError("RGB must be a bool or boolean array of length 3.")
        for checkbox, state in zip(self.RGB_checkboxes, RGB):
            checkbox.setChecked(state)

    @property
    def LUT_slider_values(self):
        """Get the current values of the LUT sliders."""
        slider_values = [slider.value() for slider in self.LUT_range_sliders]

        return slider_values

    @LUT_slider_values.setter
    def LUT_slider_values(self, bounds):
        for slider, bound in zip(self.LUT_range_sliders, bounds):
            if bound[0] == bound[1]:  # prevent division by zero
                bound = (0, 1)
            slider.blockSignals(True)
            slider.setValue(tuple(bound))
            slider.blockSignals(False)
        self.main_window._set_LUTs()

    def set_LUT_slider_ranges(self, ranges):
        for slider, slider_range in zip(self.LUT_range_sliders, ranges):
            slider.blockSignals(True)
            slider.setRange(*slider_range)
            slider.blockSignals(False)

    @property
    def _visual_settings(self):
        # retrieve the current visual settings
        out = {
            "RGB": self.RGB_visible,
            "normalize_type": self.normalize_type,
            "masks": self.masks_checkbox.isChecked(),
            "outlines": self.outlines_checkbox.isChecked(),
            "LUTs": self.LUT_slider_values,
        }
        return out

    @_visual_settings.setter
    def _visual_settings(self, settings):
        if settings["RGB"] is not None:  # RGB
            self.RGB_visible = settings["RGB"]
        self.normalize_type = settings["normalize_type"]
        self.masks_checkbox.setChecked(settings["masks"])
        self.outlines_checkbox.setChecked(settings["outlines"])
        if settings["normalize_type"] == "lut" and settings["LUTs"] is not None:  # LUT
            self.LUT_slider_values = settings["LUTs"]

    @property
    def segmentation_channels(self):
        return self.membrane_channel.currentIndex(), self.nuclear_channel.currentIndex()

    @property
    def mitosis_params(self):
        out = []
        for config, default in zip(self.mitosis_inputs, (1.5, 1, 1, 1, 1, 1, 1)):
            text = config.text()
            if text == "":
                out.append(default)
            else:
                out.append(float(text))

        return out[0], out[1], out[2:]


class RightToolbar:
    def __init__(self, parent: QMainWindow):
        self.main_window = parent
        self.last_stat_tab = 0
        self.stat_LUT_type = "frame"
        self.setup_ui()

    def setup_ui(self):
        self.scroll_area = self._create_scroll_area()
        self._connect_signals()
        return self.scroll_area

    def _create_scroll_area(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred
        )
        scroll_area.setMinimumWidth(250)

        # Main layout using QSplitter
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.setContentsMargins(5, 10, 10, 10)

        # Add main components
        main_splitter.addWidget(self._create_stat_tabs())
        main_splitter.addWidget(self._create_stat_overlay())
        main_splitter.addWidget(self._create_cell_id_widget())
        main_splitter.setSizes([200, 300, 200])

        scroll_area.setWidget(main_splitter)
        return scroll_area

    def _create_stat_tabs(self):
        self.stat_tabs = QTabWidget()
        tabs = [
            ("Histogram", self.main_window._get_histogram_tab()),
            ("Particle", self.main_window._get_particle_stat_tab()),
            ("Time Series", self.main_window._get_time_series_tab()),
        ]

        for name, widget in tabs:
            self.stat_tabs.addTab(widget, name)

        return self.stat_tabs

    def _create_stat_overlay(self):
        widget = QWidget(objectName="bordered")
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Segmentation overlay section
        layout.addLayout(self._create_seg_overlay())

        # Normalize section
        layout.addWidget(QLabel("Overlay LUTs:", self.main_window))
        layout.addWidget(self._create_normalize_widget())

        # Slider section
        slider_layout, self.stat_LUT_slider, self.stat_range_labels = (
            labeled_LUT_slider(default_range=(0, 255), parent=self)
        )
        layout.addLayout(slider_layout)

        return widget

    def _create_seg_overlay(self):
        layout = QHBoxLayout()

        self.seg_overlay_label = QLabel("Overlay Statistic:", self.main_window)
        self.seg_overlay_attr = CustomComboBox(self.main_window)
        self.seg_overlay_attr.addItems(["Select Cell Attribute"])

        layout.addWidget(self.seg_overlay_label)
        layout.addWidget(self.seg_overlay_attr)

        return layout

    def _create_normalize_widget(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.stat_frame_button = QRadioButton("Frame", self.main_window)
        self.stat_stack_button = QRadioButton("Stack", self.main_window)
        self.stat_custom_button = QRadioButton("LUT", self.main_window)

        for button in (
            self.stat_frame_button,
            self.stat_stack_button,
            self.stat_custom_button,
        ):
            layout.addWidget(button)

        self.stat_frame_button.setChecked(True)
        return widget

    def _create_cell_id_widget(self):
        widget = QWidget(objectName="bordered")
        self.cell_ID_layout = QFormLayout(widget)

        # Cell ID input
        self.selected_cell_prompt = self._create_integer_input()
        self.cell_ID_layout.addRow("Cell ID:", self.selected_cell_prompt)

        # Tracking ID input
        self.selected_particle_prompt = self._create_integer_input()
        self.cell_ID_layout.addRow("Tracking ID:", self.selected_particle_prompt)

        # Properties label
        self.cell_properties_label = QLabel(self.main_window)
        self.cell_ID_layout.addRow(self.cell_properties_label)

        return widget

    def _create_integer_input(self):
        input_field = QLineEdit(self.main_window, placeholderText="None")
        input_field.setValidator(QIntValidator(bottom=0))  # non-negative integers only
        return input_field

    def _connect_signals(self):
        # Tab switching
        self.stat_tabs.currentChanged.connect(self.main_window._stat_tab_switched)

        # Cell selection signals
        for prompt, handler in [
            (self.selected_cell_prompt, self.main_window._cell_prompt_changed),
            (self.selected_particle_prompt, self.main_window._particle_prompt_changed),
        ]:
            prompt.textChanged.connect(handler)
            prompt.returnPressed.connect(handler)

        # Stat overlay signals
        self.seg_overlay_attr.dropdownOpened.connect(
            self.main_window._get_overlay_attrs
        )
        self.seg_overlay_attr.activated.connect(self.main_window._new_seg_overlay)
        self.seg_overlay_attr.currentIndexChanged.connect(
            self.main_window._new_seg_overlay
        )

        # LUT signals
        self.stat_LUT_slider.valueChanged.connect(
            self.main_window._stat_LUT_slider_changed
        )
        for button in (
            self.stat_frame_button,
            self.stat_stack_button,
            self.stat_custom_button,
        ):
            button.toggled.connect(self.main_window._update_stat_LUT)


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)

        # If it's a widget, delete it
        if item.widget():
            item.widget().deleteLater()
        # If it's a layout, clear and delete it recursively
        elif item.layout():
            clear_layout(item.layout())
            item.layout().deleteLater()
        # If it's a spacer, just remove it (no need to delete)
        else:
            del item


def labeled_LUT_slider(
    slider_name=None, default_range=(0, 65535), parent=None, digit_width=5
):
    labels_and_slider = QHBoxLayout()
    labels_and_slider.setSpacing(2)
    if slider_name is not None:
        slider_label = QLabel(slider_name)
        labels_and_slider.addWidget(slider_label)

    slider = FineScrubQRangeSlider(orientation=Qt.Orientation.Horizontal, parent=parent)
    slider.setRange(*default_range)
    slider.setValue(default_range)

    range_labels = [QLineEdit(str(val)) for val in slider.value()]
    for label in range_labels:
        label.setFixedWidth(digit_width * 6)
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
        min_val = int(range_labels[0].text())
        max_val = int(range_labels[1].text())

        if min_val < slider.minimum():
            slider.setMinimum(min_val)
        elif min_val > max_val:
            min_val = max_val
            range_labels[0].setText(str(min_val))
        slider.setValue((min_val, max_val))

    def update_max_slider_from_edit():
        min_val = int(range_labels[0].text())
        max_val = int(range_labels[1].text())

        if max_val > slider.maximum():
            slider.setMaximum(max_val)
        elif max_val < min_val:
            max_val = min_val
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
    labels_and_slider.addSpacerItem(
        QSpacerItem(10, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    )
    labels_and_slider.addWidget(slider)
    labels_and_slider.addSpacerItem(
        QSpacerItem(10, 0, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    )
    labels_and_slider.addWidget(range_labels[1])

    return labels_and_slider, slider, range_labels
