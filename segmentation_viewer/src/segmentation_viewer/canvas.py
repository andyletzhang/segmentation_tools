from multiprocessing import cpu_count

import cv2
import fastremap
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QBrush, QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsPathItem, QGraphicsPolygonItem, QGraphicsScene, QHBoxLayout, QWidget
from segmentation_tools.shape_operations import get_bounding_box, get_enclosed_pixels, get_mask_boundary

from .workers import MaskProcessor

debug_execution_times = False

N_CORES = cpu_count()


class PyQtGraphCanvas(QWidget):
    def __init__(self, parent=None, cell_n_colors=10, cell_cmap='tab10'):
        from matplotlib import colormaps

        super().__init__(parent)
        self.main_window = parent
        self.selected_cell_color = 'white'
        self.selected_cell_alpha = 0.3
        self.outlines_color = 'white'
        self.outlines_alpha = 0.5
        self.masks_alpha = 0.5
        self.mask_processor = MaskProcessor(self, n_cores=2)

        # cell mask colors
        self.cell_n_colors = cell_n_colors
        self.cell_cmap = colormaps.get_cmap(cell_cmap)

        # Create a layout for the widget
        plot_layout = QHBoxLayout(self)
        plot_layout.setSpacing(1)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        # Create a PlotWidget for the image and segmentation views
        self.img_plot = SegPlot(title='Image', border='w', parent=self)
        self.seg_plot = SegPlot(title='Segmentation', border='w', parent=self)

        plot_layout.addWidget(self.img_plot)
        plot_layout.addWidget(self.seg_plot)

        # Initialize data - 512x512 with white outline
        self.img_data = np.ones((512, 512, 3), dtype=np.uint8)
        self.img_data[5:-5, 5:-5] = 0  # white border
        self.seg_data = self.img_data.copy()

        # Plot the data
        self.img = RGB_ImageItem(img_data=self.img_data, plot=self.img_plot, parent=self)
        self.seg = pg.ImageItem(self.seg_data)
        self.img_outline_overlay = pg.ImageItem()
        self.seg_stat_overlay = pg.ImageItem()
        self.cb = pg.ColorBarItem(interactive=False, orientation='horizontal', width=15)
        self.update_stat_overlay_lut('viridis')
        self.update_img_outline_lut()
        self.seg.setLookupTable(transparent_binary_lut((1,1,1), alpha=1))
        self.cb.setFixedWidth(100)
        self.cb.setImageItem(self.seg_stat_overlay)
        self.cb.setVisible(False)
        self.seg_plot.scene().addItem(self.cb)
        # Position relative to plot edges
        self.cb.setPos(self.seg_plot.width() - 40, 50)

        self.mask_overlay = [pg.ImageItem(), pg.ImageItem()]
        self.selection_overlay = [pg.ImageItem(), pg.ImageItem()]
        self.FUCCI_overlay = [pg.ImageItem(), pg.ImageItem()]
        self.tracking_overlay = [pg.ImageItem(), pg.ImageItem()]
        self.mitosis_overlay = [pg.ImageItem(), pg.ImageItem()]

        # add images to the plots
        # self.img_plot.addItem(self.img)
        self.img_plot.addItem(self.img_outline_overlay)

        self.img_plot.addItem(self.mask_overlay[0])
        self.seg_plot.addItem(self.mask_overlay[1])

        self.seg_plot.addItem(self.seg_stat_overlay)

        self.img_plot.addItem(self.tracking_overlay[0])
        self.seg_plot.addItem(self.tracking_overlay[1])

        self.img_plot.addItem(self.FUCCI_overlay[0])
        self.seg_plot.addItem(self.FUCCI_overlay[1])

        self.img_plot.addItem(self.selection_overlay[0])
        self.seg_plot.addItem(self.selection_overlay[1])

        self.img_plot.addItem(self.mitosis_overlay[0])
        self.seg_plot.addItem(self.mitosis_overlay[1])

        self.seg_plot.addItem(self.seg)

        # Set initial zoom levels
        self.img_plot.setRange(xRange=[0, self.img_data.shape[1]], yRange=[0, self.img_data.shape[0]], padding=0)
        self.seg_plot.setRange(xRange=[0, self.seg_data.shape[1]], yRange=[0, self.seg_data.shape[0]], padding=0)

        # Connect the range change signals to the custom slots
        self.img_plot.sigRangeChanged.connect(self.sync_seg_plot)
        self.seg_plot.sigRangeChanged.connect(self.sync_img_plot)
        # Connect the mouse move signals to the custom slots
        self.img_mouse_proxy = pg.SignalProxy(self.img_plot.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)
        self.seg_mouse_proxy = pg.SignalProxy(self.seg_plot.scene().sigMouseMoved, rateLimit=60, slot=self.mouse_moved)

        # Create crosshair lines
        self.img_vline = pg.InfiniteLine(angle=90, movable=False)
        self.img_hline = pg.InfiniteLine(angle=0, movable=False)
        self.seg_vline = pg.InfiniteLine(angle=90, movable=False)
        self.seg_hline = pg.InfiniteLine(angle=0, movable=False)
        self.img_plot.addItem(self.img_vline, ignoreBounds=True)
        self.img_plot.addItem(self.img_hline, ignoreBounds=True)
        self.seg_plot.addItem(self.seg_vline, ignoreBounds=True)
        self.seg_plot.addItem(self.seg_hline, ignoreBounds=True)

    def hide_crosshairs(self):
        for line in [self.img_vline, self.img_hline, self.seg_vline, self.seg_hline]:
            line.setVisible(False)

    def show_crosshairs(self):
        for line in [self.img_vline, self.img_hline, self.seg_vline, self.seg_hline]:
            line.setVisible(True)

    def wheelEvent(self, event):
        """Redirect wheel events to the main window."""
        self.main_window._canvas_wheelEvent(event)

    def update_outlines(self):
        """Update outlines overlays to match the current frame."""
        self.seg_data = self.image_transform(self.main_window.frame.outlines)
        self.seg.setImage(self.seg_data, levels=(0, 1))
        self.img_outline_overlay.setImage(self.seg_data, levels=(0,1))

    def overlay_outlines(self):
        self.img_outline_overlay.setVisible(self.main_window.outlines_visible)


    def overlay_masks(self):
        for layer in self.mask_overlay:
            layer.setVisible(self.main_window.masks_visible)

        if not hasattr(self.main_window.frame, 'stored_mask_overlay'):
            self.draw_masks()

        for layer, overlay in zip(self.mask_overlay, self.main_window.frame.stored_mask_overlay):
            layer.setImage(overlay)

    def toggle_RGB_checks(self, RGB_checks):
        self.img.toggle_RGB_checks(RGB_checks)

    def random_cell_color(self, n=0):
        random_colors = self.cell_cmap(self.random_color_ID(n))

        if n == 0:
            return random_colors[:3]
        else:
            return random_colors[:, :3]

    def random_color_ID(self, n: int = 0, ignore: int | list[int] | None = None):
        cell_colors = np.arange(self.cell_n_colors)
        if ignore is not None:
            cell_colors = np.setdiff1d(cell_colors, ignore)

        random_IDs = np.random.randint(0, len(cell_colors), size=max(1, n))
        random_IDs = cell_colors[random_IDs]

        if n == 0:
            return random_IDs[0]
        else:
            return random_IDs

    def draw_masks(self, alpha=None, frame=None):
        if alpha is None:
            alpha = self.masks_alpha
        # get cell colors
        if frame is None:
            frame = self.main_window.frame
        if frame.n_cells == 0:
            img_masks, seg_masks = np.zeros((*frame.masks.shape, 4)), np.zeros((*frame.masks.shape, 4))
        else:
            try:
                cell_colors = frame.get_cell_attrs('color_ID')  # retrieve the stored colors for each cell
            except AttributeError:
                # from monolayer_tracking.networks import color_masks, greedy_color # generate pseudo-random colors
                # random_colors=color_masks(self.main_window.frame.masks)
                cell_colors = self.random_cell_color(frame.n_cells)
                self.main_window.frame.set_cell_attr('color_ID', cell_colors)

            # highlight all cells with the specified colors
            cell_indices = fastremap.unique(frame.masks)[1:] - 1
            img_masks, seg_masks = self.highlight_cells(
                cell_indices, frame=frame, alpha=alpha, cell_colors=cell_colors, layer='mask'
            )

        frame.stored_mask_overlay = [img_masks, seg_masks]

    def draw_masks_parallel(self, frames=None):
        """Process multiple frames in parallel using QThreadPool."""
        if frames is None:
            frames = self.main_window.stack.frames

        if N_CORES == 1:
            return  # No background loading if only one core is available
        else:
            self.mask_processor.draw_masks_parallel(frames)

    def draw_masks_bg(self, frame):
        """Process a single frame in the background."""
        if N_CORES == 1:
            return
        else:
            self.mask_processor.add_frame_task(frame)

    def clear_overlay(self, overlay):
        overlay = getattr(self, f'{overlay}_overlay')
        overlay[0].clear()
        overlay[1].clear()

    def highlight_cells(
        self,
        cell_indices,
        frame=None,
        layer='selection',
        alpha=None,
        color=None,
        cell_colors=None,
        img_type='masks',
        seg_type='masks',
    ):
        from matplotlib.colors import to_rgb

        if alpha is None:
            alpha = self.selected_cell_alpha
        if color is None:
            color = self.selected_cell_color

        if frame is None:
            frame = self.main_window.frame
        elif isinstance(frame, int):
            frame = self.main_window.stack.frames[frame]

        if frame == self.main_window.frame:
            drawing_layers = True

        else:
            if layer != 'mask':
                Warning(
                    f'Only mask overlay can be drawn on stored frames, but highlight_cells was called with layer {layer}. Proceeding with mask overlay.'
                )
            layer = 'mask'
            drawing_layers = False

        masks = frame.masks

        if cell_colors is None:  # single color mode
            if np.issubdtype(type(color), np.integer):
                color = self.cell_cmap(color)
            color = [*to_rgb(color), alpha]  # convert color to RGBA
            mask_overlay = np.isin(masks - 1, cell_indices)[..., np.newaxis] * color

        else:  # multi-color mode
            for n in range(len(cell_colors)):
                color = cell_colors[n]
                if np.issubdtype(type(color), np.integer):
                    cell_colors[n] = self.cell_cmap(color)
            num_labels = masks.max() + 1  # number of unique labels (including background)
            mask_overlay = np.zeros((*masks.shape, 4))
            cell_colors = np.array([[*to_rgb(c), alpha] for c in cell_colors])
            color_map = np.zeros((num_labels, 4))
            for i, cell_index in enumerate(cell_indices):
                color_map[cell_index + 1] = cell_colors[i]
            mask_overlay = color_map[masks]

        opaque_mask = mask_overlay.copy()
        opaque_mask[mask_overlay[..., -1] != 0, -1] = 1
        if img_type == 'outlines':
            mask_overlay[frame.outlines == 0] = 0
        if seg_type == 'outlines':
            opaque_mask[frame.outlines == 0] = 0

        mask_overlay = self.image_transform(mask_overlay)
        opaque_mask = self.image_transform(opaque_mask)

        if drawing_layers:
            layer_overlay = getattr(self, f'{layer}_overlay')
            layer_overlay[0].setImage(mask_overlay)
            layer_overlay[1].setImage(opaque_mask)

        return mask_overlay, opaque_mask

    def redraw_cell_mask(self, cell, **highlight_kwargs):
        if not hasattr(self.main_window.stack.frames[cell.frame], 'stored_mask_overlay'):
            return  # no mask overlay to update
        highlight_kwargs = {'color': cell.color_ID, 'layer': 'mask', 'mode': 'overwrite'} | highlight_kwargs
        self.add_cell_highlight(cell.n, frame=cell.frame, **highlight_kwargs)

    def add_cell_highlight(
        self,
        cell_index: int,
        frame=None,
        layer: str = 'selection',
        alpha: float | None = None,
        color: int | None = None,
        img_type: str = 'masks',
        seg_type: str = 'masks',
        seg_alpha: bool | float = False,
        mode: str = 'overwrite',
    ):
        from matplotlib.colors import to_rgb

        if frame is None:
            frame = self.main_window.frame
        elif isinstance(frame, int):
            frame = self.main_window.stack.frames[frame]

        if alpha is None:
            if layer == 'mask':
                alpha = self.masks_alpha
            elif layer == 'selection':
                alpha = self.selected_cell_alpha
            else:
                raise ValueError(f'Must pass alpha to layer {layer}')
        if color is None:
            if layer == 'selection':
                color = self.selected_cell_color
            else:
                color = self.random_color_ID()

        # get the binarized mask for the specified cell
        cell_mask = self.image_transform(frame.masks == cell_index + 1)

        if frame == self.main_window.frame:
            drawing_layers = True
            layer_overlay = getattr(self, f'{layer}_overlay')
            for layer_item in layer_overlay:
                if layer_item.image is None:  # initialize the overlay
                    layer_item.image = np.zeros(cell_mask.shape + (4,))
            overlays = [layer_overlay[0].image, layer_overlay[1].image]
        else:
            if layer != 'mask':
                Warning(
                    f'Only mask overlay can be drawn on stored frames, but add_cell_highlight was called with layer {layer}. Proceeding with mask overlay.'
                )
            layer = 'mask'
            drawing_layers = False
            if hasattr(frame, 'stored_mask_overlay'):
                overlays = frame.stored_mask_overlay
            else:
                Warning(f'No stored mask overlay found for frame {frame.frame_number}. No cell highlight will be drawn.')
                return

        if isinstance(color, str) and color == 'none':
            # remove the cell from the overlay
            overlays[0][cell_mask] = 0
            overlays[1][cell_mask] = 0
        else:
            if np.issubdtype(type(color), np.integer):
                color = self.cell_cmap(color)

            # Convert color to RGBA
            color = [*to_rgb(color), alpha]

            # get bounding box of cell mask
            xmin, xmax, ymin, ymax = get_bounding_box(cell_mask)
            if xmin is None:
                raise IndexError(f'No pixels found for cell number {cell_index}.')

            cell_mask_bbox = cell_mask[xmin:xmax, ymin:ymax]
            # Get the mask for the specified cell
            img_cell_mask = cell_mask_bbox[..., np.newaxis] * color
            # Create an opaque mask for the specified cell
            seg_cell_mask = img_cell_mask.copy()

            if isinstance(seg_alpha, float):  # custom segmentation alpha
                seg_cell_mask[cell_mask_bbox, -1] = seg_alpha
            elif not seg_alpha:  # default: opaque
                seg_cell_mask[cell_mask_bbox, -1] = 1
            else:  # default: same as img alpha
                pass

            if img_type == 'outlines' or seg_type == 'outlines':
                outlines = get_mask_boundary(cell_mask_bbox)
                if img_type == 'outlines':
                    img_cell_mask[~outlines] = 0
                if seg_type == 'outlines':
                    seg_cell_mask[~outlines] = 0

            # Add the cell mask to the existing overlay
            img_out = img_cell_mask[cell_mask_bbox]
            seg_out = seg_cell_mask[cell_mask_bbox]
            for out, overlay in zip([img_out, seg_out], overlays):
                if mode == 'overwrite':
                    overlay[xmin:xmax, ymin:ymax][cell_mask_bbox] = out
                elif mode == 'add':
                    overlay[xmin:xmax, ymin:ymax][cell_mask_bbox] += out
                elif mode == 'blend':
                    current = overlay[xmin:xmax, ymin:ymax][cell_mask_bbox]
                    alpha_out = out[..., -1] + current[..., -1] - out[..., -1] * current[..., -1]
                    zero_alpha = alpha_out == 0
                    safe_alpha = alpha_out.copy()
                    safe_alpha[zero_alpha] = 1
                    out[..., :-1] = (
                        out[..., :-1] * out[..., -1, np.newaxis]
                        + current[..., :-1] * current[..., -1, np.newaxis] * (1 - out[..., -1, np.newaxis])
                    ) / safe_alpha[..., np.newaxis]
                    out[..., -1] = alpha_out
                    out[zero_alpha] = 0

                    overlay[xmin:xmax, ymin:ymax][cell_mask_bbox] = out

        # Update the overlay images
        if drawing_layers:
            layer_overlay[0].setImage(overlays[0])
            layer_overlay[1].setImage(overlays[1])

        # store mask overlays if layer is mask
        if layer == 'mask':
            frame.stored_mask_overlay = overlays

        return

    def update_img_outline_lut(self):
        self.img_outline_overlay.setLookupTable(transparent_binary_lut(self.outlines_color, self.outlines_alpha))

    def update_stat_overlay_lut(self, colormap):
        lut = get_matplotlib_LUT(colormap)
        self.seg_stat_overlay.setLookupTable(lut)
        self.cb.setColorMap(pg.ColorMap(pos=None, color=lut))
        self.seg_stat_overlay.current_cmap = colormap

    def get_plot_coords(self, pos=None, pixels=True):
        """Get the pixel coordinates of the mouse cursor."""
        if pos is None:
            pos = self.mapFromGlobal(QCursor.pos())
            pos = QPointF(pos)
        mouse_point = self.img_plot.plotItem.vb.mapSceneToView(
            pos
        )  # axes are the same for both plots so we can use either to transform
        x, y = mouse_point.x(), mouse_point.y()
        if pixels:
            x, y = int(y), int(x)
        return x, y

    def mouse_moved(self, event):
        pos = event[0]
        x, y = self.get_plot_coords(pos, pixels=False)
        self.update_cursor(x, y)

        self.main_window._mouse_moved(pos)

    def update_cursor(self, x, y):
        """Update the segmentation plot cursor based on the image plot cursor."""
        self.seg_vline.setPos(x)
        self.seg_hline.setPos(y)
        self.img_vline.setPos(x)
        self.img_hline.setPos(y)
        self.main_window._update_coordinate_label(int(x), int(y))

    def save_img_plot(self, filename):
        img = self.draw_plot(plot='img')
        img.save(filename, compress_level=0, format='png')

    def save_seg_plot(self, filename):
        img = self.draw_plot(plot='seg')
        img.save(filename, compress_level=0, format='png')

    def draw_plot(self, plot: str = 'img'):
        """Take a screenshot of the image plot."""
        if plot == 'img':
            plot = self.img_plot
        elif plot == 'seg':
            plot = self.seg_plot
        else:
            raise ValueError(f'Invalid plot type: {plot}')

        image_shape = self.img_data.shape[:2]
        images = plot.get_image_items()
        # RGBA composition
        RGBA_images = []
        for img in reversed(images):
            if not img.isVisible():  # only draw enabled layers
                continue
            if img.image.shape[:2] != image_shape:  # skip if the image is not the same size as the main image
                continue

            img_rgba = img_item_to_RGBA(img)
            RGBA_images.append(self.inverse_image_transform(img_rgba))  # remove visual transformation
        if len(RGBA_images) == 0:
            return None
        return composite_images_pillow(RGBA_images)

    @property
    def cursor_pixels(self):
        """Get the cursor position as integers."""
        return self.get_plot_coords(pixels=True)

    @property
    def cursor_pos(self):
        """Get the exact cursor position."""
        return self.get_plot_coords(pixels=False)

    @property
    def is_inverted(self):
        return self.main_window.is_inverted

    def update_display(self, img_data=None, seg_data=None, RGB_checks=None):
        import time

        execution_times = {}
        start_time = time.time()
        if img_data is None:
            img_data = self.img_data
        execution_times['if img_data is None: img_data = self.img_data'] = time.time() - start_time

        start_time = time.time()
        if seg_data is None:
            seg_data = self.seg_data
        execution_times['if seg_data is None: seg_data = self.seg_data'] = time.time() - start_time

        start_time = time.time()
        self.img_data = img_data.copy()
        execution_times['self.img_data = img_data.copy()'] = time.time() - start_time

        start_time = time.time()
        self.seg_data = seg_data.copy()
        execution_times['self.seg_data = seg_data.copy()'] = time.time() - start_time

        start_time = time.time()
        self.img.setImage(self.img_data)
        execution_times['self.img.setImage(self.img_data)'] = time.time() - start_time

        start_time = time.time()
        self.seg.setImage(self.seg_data, levels=(0, 1))
        self.img_outline_overlay.setImage(self.seg_data, levels=(0,1))
        execution_times['self.seg.setImage(self.seg_data)'] = time.time() - start_time

        start_time = time.time()
        self.overlay_masks()
        execution_times['self.overlay_masks()'] = time.time() - start_time

        # Print all execution times sorted by duration
        if debug_execution_times:
            print('-----------UPDATE DISPLAY TIMES-----------')
            sorted_execution_times = sorted(execution_times.items(), key=lambda item: item[1], reverse=True)
            for description, duration in sorted_execution_times:
                if duration < 0.001:
                    continue
                print(f'{description}: {duration:.4f} seconds')

    def image_transform(self, img_data):
        return np.fliplr(np.rot90(img_data, 3))

    def inverse_image_transform(self, img_data):
        return np.rot90(np.fliplr(img_data), 1)

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

    def close(self):
        if hasattr(self, 'mask_processor'):
            self.mask_processor.abort_all_tasks()

def transparent_binary_lut(color, alpha):
    from matplotlib.colors import to_rgb

    color = np.array([*to_rgb(color), alpha])  # convert color to RGBA
    lut = np.zeros((2, 4), dtype=np.uint8)
    lut[1] = (color * 255).astype(np.uint8)
    return lut

# util functions
def img_item_to_RGBA(img: pg.ImageItem) -> np.ndarray:
    # apply levels
    image = img.image.copy()
    nan_mask = np.isnan(image)
    image[nan_mask] = 0
    if img.levels is not None:
        levels = img.levels
        image = np.clip(image, levels[0], levels[1])
        image = (image - levels[0]) / (levels[1] - levels[0]) * 255

    image = image.astype(np.uint8)
    # apply LUT if necessary
    if img.lut is not None:
        image = img.lut[image]

    if image.ndim == 2:
        image = image[..., np.newaxis] * np.ones((1, 1, 3), dtype=image.dtype)

    # add alpha channel
    if image.shape[-1] == 3:
        alpha = np.ones((*image.shape[:2], 1), dtype=np.uint8) * 255
        image = np.concatenate((image, alpha), axis=-1)

    image[nan_mask] = 0

    return image


def composite_images_pillow(images):
    from PIL import Image

    """Composites a list of PIL RGBA images using Pillow."""
    base = Image.fromarray(images[0], mode='RGBA')
    for img in images[1:]:
        base = Image.alpha_composite(base, Image.fromarray(img, mode='RGBA'))
    return base


class SegPlot(pg.PlotWidget):
    """
    Custom PlotWidget for the segmentation viewer.
    Only functional difference is to redirect mouse wheel events to the main window.
    """

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = parent
        self.setMenuEnabled(False)
        self.getViewBox().invertY(True)
        self.setAspectLocked(True)
        self.setContentsMargins(0, 0, 0, 0)

    def wheelEvent(self, event):
        # ctrl+wheel zooms like normal, otherwise send it up to the main window
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(event)  # PlotWidget handles zooming
        else:
            event.ignore()

    def get_image_items(self):
        return [item for item in self.items() if isinstance(item, pg.ImageItem) and item.image is not None]


class RGB_ImageItem:
    def __init__(self, plot: pg.PlotWidget, img_data=None, parent=None):
        self.canvas = parent
        if img_data is None:
            img_data = np.zeros((512, 512, 3), dtype=np.uint8)

        self.scene = QGraphicsScene()
        self.img_data = img_data
        self.red = pg.ImageItem(self.img_data[..., 0])
        self.green = pg.ImageItem(self.img_data[..., 1])
        self.blue = pg.ImageItem(self.img_data[..., 2])
        self.channels = [self.red, self.green, self.blue]

        # Add items to the view
        for item in self.channels:
            self.scene.addItem(item)
            item.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)

        # default LUTs
        self.LUT_options = {
            'Grays': (255, 255, 255),
            'Reds': (255, 0, 0),
            'Greens': (0, 255, 0),
            'Blues': (0, 0, 255),
            'Yellows': (255, 255, 0),
            'Cyans': (0, 255, 255),
            'Magentas': (255, 0, 255),
        }
        self.LUTs = ('Reds', 'Greens', 'Grays')
        self.setLookupTable('RGB')

        self.img_item = pg.ImageItem(self.image(), levels=(0, 255))
        plot.addItem(self.img_item)
        self.show_grayscale = False
        self.update_LUTs()

    def toggle_RGB_checks(self, RGB_checks):
        for i, check in enumerate(RGB_checks):
            self.channels[i].setVisible(check)
        self.refresh()

    def image(self):
        """Get the rendered image from the specified plot."""
        width, height = self.red.image.shape
        output_img = QImage(width, height, QImage.Format.Format_RGB32)
        output_img.fill(0)
        painter = QPainter(output_img)
        self.scene.render(painter)
        painter.end()
        ptr = output_img.bits()

        ptr.setsize(output_img.sizeInBytes())
        composite_array = np.array(ptr).reshape((height, width, 4))  # Format_RGB32 includes alpha
        rgb_array = composite_array[..., :3][..., ::-1]

        if self.canvas.is_inverted:
            rgb_array = inverted_contrast(rgb_array)

        # add alpha channel
        alpha = np.ones((height, width, 1), dtype=np.uint8) * 255
        rgb_array = np.concatenate((rgb_array, alpha), axis=-1)
        return rgb_array

    def refresh(self):
        """Re-render the image item."""
        self.img_item.setImage(self.image(), autoLevels=False)

    def setImage(self, img_data):
        self.img_data = img_data
        if img_data.ndim == 2:
            self.red.setImage(self.img_data, autoLevels=False)
            self.green.clear()
            self.blue.clear()
            self.setLookupTable('gray')
        else:  # RGB image
            if self.img_data.shape[-1] == 2:  # two channels--assume RG and convert to RGB
                self.img_data = np.concatenate((self.img_data, np.zeros((*self.img_data.shape[:-1], 1), dtype=np.uint8)), axis=-1)
            self.red.setImage(self.img_data[..., 0], autoLevels=False)
            self.green.setImage(self.img_data[..., 1], autoLevels=False)
            self.blue.setImage(self.img_data[..., 2], autoLevels=False)
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        self.refresh()

    def update_LUTs(self):
        if self.show_grayscale:
            self.setLookupTable('gray')
        else:
            self.setLookupTable('RGB')
        self.refresh()

    def setLevels(self, levels, refresh=True):
        """Update the levels of the image items based on the sliders."""
        for level, item in zip(levels, self.channels):
            item.setLevels(level)
        if refresh:
            self.refresh()

    def getLevels(self):
        """Get the levels of the image items."""
        return [item.getLevels() for item in self.channels]

    def create_lut(self, color):
        lut = np.zeros((256, 3), dtype=np.ubyte)
        for i in range(256):
            lut[i] = [color.red() * i // 255, color.green() * i // 255, color.blue() * i // 255]
        return lut

    def setLookupTable(self, lut_style):
        """Set the lookup table for the image items."""
        if lut_style == 'gray':
            luts = self.gray_lut()
        elif lut_style == 'RGB':
            luts = self.RGB_lut()

        for item, lut in zip([self.red, self.green, self.blue], luts):
            item.setLookupTable(lut)

    def gray_lut(self):
        return [self.create_lut(QColor(255, 255, 255))] * 3

    def RGB_lut(self):
        return [self.create_lut(QColor(*self.LUT_options[lut])) for lut in self.LUTs]

    def set_grayscale(self, grayscale):
        self.show_grayscale = grayscale
        self.update_LUTs()


class CellMaskPolygons:
    """
    pair of CellMaskPolygon objects for image and segmentation plots
    """

    def __init__(self, parent=None, *args, **kwargs):
        self.canvas = parent
        self.img_poly = CellMaskPolygon(*args, **kwargs)
        self.seg_poly = CellMaskPolygon(*args, **kwargs)

    # all methods are passed to the corresponding CellMaskPolygon objects
    def clearPoints(self):
        self.img_poly.clearPoints()
        self.seg_poly.clearPoints()

    def update_polygon(self):
        self.img_poly.update_polygon()
        self.seg_poly.update_polygon()

    def add_vertex(self, y, x):
        self.img_poly.add_vertex(y, x)
        self.seg_poly.add_vertex(y, x)

    @property
    def points(self):
        return self.img_poly.points

    def enclosed_pixels(self):
        return self.img_poly.enclosed_pixels()


class CellMaskPolygon(QGraphicsPolygonItem):
    """Polygonal overlay for drawing the current mask."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPen(QPen(QColor(255, 255, 255, 50)))
        self.setBrush(QBrush(QColor(120, 120, 120, 50)))
        self.points = []

    def clearPoints(self):
        self.points = []
        self.update_polygon()

    def update_polygon(self):
        polygon = QPolygonF(self.points)
        self.setPolygon(polygon)

    def add_vertex(self, y, x):
        y, x = y + 0.5, x + 0.5
        self.points.append(QPointF(y, x))
        self.update_polygon()
        self.last_handle_pos = (y, x)

    def enclosed_pixels(self) -> np.ndarray:
        """Return all pixels enclosed by the polygon."""

        points = np.array([(p.x(), p.y() - 0.5) for p in self.points])
        points = np.vstack((points, points[0]))  # close the polygon

        return get_enclosed_pixels(points)


class CellSplitLines:
    """
    pair of CellSplitLine objects for image and segmentation plots
    """

    def __init__(self, parent=None, *args, **kwargs):
        self.canvas = parent
        self.img_line = CellSplitLine(*args, **kwargs)
        self.seg_line = CellSplitLine(*args, **kwargs)

    # all methods are passed to the corresponding CellSplitLine objects
    def clearPoints(self):
        self.img_line.clearPoints()
        self.seg_line.clearPoints()

    def update_line(self):
        self.img_line.update_line()
        self.seg_line.update_line()

    def add_vertex(self, y, x):
        self.img_line.add_vertex(y, x)
        self.seg_line.add_vertex(y, x)

    @property
    def points(self):
        return self.img_line.points


class CellSplitLine(QGraphicsPathItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPen(QPen(QColor(255, 255, 255, 100)))
        self.points = []

    def clearPoints(self):
        self.points = []
        self.update_line()

    def update_line(self):
        path = QPainterPath()
        if self.points:
            path.moveTo(self.points[0])
            for point in self.points[1:]:
                path.lineTo(point)
        self.setPath(path)

    def add_vertex(self, y, x):
        y, x = y + 0.5, x + 0.5
        self.points.append(QPointF(y, x))
        self.update_line()
        self.last_handle_pos = (y, x)


def get_matplotlib_LUT(name):
    from matplotlib import cm

    colormap = cm.get_cmap(name)
    lut = (colormap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
    return lut


def inverted_contrast(img):
    lut = np.array([255 - i for i in range(256)], dtype=np.uint8)  # Lookup table
    img_inv = cv2.LUT(img, lut)  # Fast inversion

    hsv = cv2.cvtColor(img_inv, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + 90) % 180  # In-place hue rotation

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
