import fastremap
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QBrush, QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PyQt6.QtWidgets import QGraphicsPathItem, QGraphicsPolygonItem, QGraphicsScene, QHBoxLayout, QWidget
from shapely.geometry import LineString
from shapely.ops import polygonize, unary_union

try:
    import cupy as xp
    from cucim.skimage.color import hsv2rgb, rgb2hsv

    on_gpu = True
except ImportError:
    import numpy as xp
    from skimage.color import hsv2rgb, rgb2hsv

    Warning('cupy and/or cucim not found. Inverted contrast may be slow.')
    on_gpu = False


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
        stat_overlay_lut = get_matplotlib_LUT('viridis')
        self.seg_stat_overlay.setLookupTable(stat_overlay_lut)
        self.cb = pg.ColorBarItem(interactive=False, orientation='horizontal', colorMap='viridis', width=15)
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

        self.seg_plot.addItem(self.seg)

        # Set initial zoom levels
        self.img_plot.setRange(xRange=[0, self.img_data.shape[1]], yRange=[0, self.img_data.shape[0]], padding=0)
        self.seg_plot.setRange(xRange=[0, self.seg_data.shape[1]], yRange=[0, self.seg_data.shape[0]], padding=0)

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

    def wheelEvent(self, event):
        """Redirect wheel events to the main window."""
        self.main_window._canvas_wheelEvent(event)

    def overlay_outlines(self):
        self.img_outline_overlay.setVisible(self.main_window.outlines_visible)

    def draw_outlines(self, color=None, alpha=None):
        """Overlay the outlines of the masks on the image plot."""
        if alpha is None:
            alpha = self.outlines_alpha
        if color is None:
            color = self.outlines_color

        from matplotlib.colors import to_rgb

        color = [*to_rgb(color), alpha]

        overlay = np.zeros((*self.main_window.frame.masks.shape, 4))
        overlay[self.main_window.frame.outlines] = color

        overlay = self.image_transform(overlay)
        self.img_outline_overlay.setImage(overlay)
        self.overlay_outlines()

    def overlay_masks(self):
        for layer in self.mask_overlay:
            layer.setVisible(self.main_window.masks_visible)

        if not hasattr(self.main_window.frame, 'stored_mask_overlay'):
            self.draw_masks()
        else:
            for layer, overlay in zip(self.mask_overlay, self.main_window.frame.stored_mask_overlay):
                layer.setImage(overlay)

    def random_cell_color(self, n=1):
        random_colors = self.cell_cmap(self.random_color_ID(n))

        if n == 1:
            return random_colors[:3]
        else:
            return random_colors[:, :3]

    def random_color_ID(self, n=1):
        random_IDs = np.random.randint(0, self.cell_n_colors, size=n)

        if n == 1:
            return random_IDs[0]
        else:
            return random_IDs

    def draw_masks(self, alpha=None):
        if alpha is None:
            alpha = self.masks_alpha
        # get cell colors
        try:
            cell_colors = self.main_window.frame.get_cell_attrs('color_ID')  # retrieve the stored colors for each cell
        except AttributeError:
            # from monolayer_tracking.networks import color_masks, greedy_color # generate pseudo-random colors
            # random_colors=color_masks(self.main_window.frame.masks)
            cell_colors = self.random_cell_color(self.main_window.frame.masks.max())
            self.main_window.frame.set_cell_attr('color_ID', cell_colors)

        # highlight all cells with the specified colors
        cell_indices = fastremap.unique(self.main_window.frame.masks)[1:] - 1
        img_masks, seg_masks = self.highlight_cells(cell_indices, alpha=alpha, cell_colors=cell_colors, layer='mask')

        return img_masks, seg_masks

    def clear_FUCCI_overlay(self):
        self.FUCCI_overlay[0].clear()
        self.FUCCI_overlay[1].clear()

    def clear_tracking_overlay(self):
        self.tracking_overlay[0].clear()
        self.tracking_overlay[1].clear()

    def highlight_cells(
        self, cell_indices, layer='selection', alpha=None, color=None, cell_colors=None, img_type='masks', seg_type='masks'
    ):
        from matplotlib.colors import to_rgb

        if alpha is None:
            alpha = self.selected_cell_alpha
        if color is None:
            color = self.selected_cell_color

        masks = self.main_window.frame.masks

        layer = getattr(
            self, f'{layer}_overlay'
        )  # get the specified overlay layer: selection for highlighting, mask for colored masks

        if cell_colors is None:  # single color mode
            color = [*to_rgb(color), alpha]  # convert color to RGBA
            mask_overlay = np.isin(masks - 1, cell_indices)[..., np.newaxis] * color

        else:  # multi-color mode
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
            mask_overlay[self.main_window.frame.outlines == 0] = 0
        if seg_type == 'outlines':
            opaque_mask[self.main_window.frame.outlines == 0] = 0

        mask_overlay = self.image_transform(mask_overlay)
        opaque_mask = self.image_transform(opaque_mask)

        layer[0].setImage(mask_overlay)
        layer[1].setImage(opaque_mask)

        # store mask overlays if layer is mask
        if layer == self.mask_overlay:
            self.main_window.frame.stored_mask_overlay = [mask_overlay, opaque_mask]

        return mask_overlay, opaque_mask

    def add_cell_highlight(
        self,
        cell_index,
        frame=None,
        layer='selection',
        alpha=None,
        color=None,
        img_type='masks',
        seg_type='masks',
        seg_alpha=False,
    ):
        from matplotlib.colors import to_rgb

        if frame is None:
            frame = self.main_window.frame
        elif isinstance(frame, int):
            frame = self.main_window.stack.frames[frame]

        if alpha is None:
            alpha = self.selected_cell_alpha
        if color is None:
            color = self.selected_cell_color

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
            if isinstance(color, int):
                color = self.cell_cmap(color)

            # Convert color to RGBA
            color = [*to_rgb(color), alpha]

            # get bounding box of cell mask
            xmin, xmax, ymin, ymax = self.get_bounding_box(cell_mask)
            if xmin is None:
                raise IndexError(f'No pixels found for cell number {cell_index}.')

            cell_mask_bbox = cell_mask[xmin:xmax, ymin:ymax]
            # Get the mask for the specified cell
            img_cell_mask = cell_mask_bbox[..., np.newaxis] * color
            # Create an opaque mask for the specified cell
            seg_cell_mask = img_cell_mask.copy()
            if not seg_alpha:
                seg_cell_mask[cell_mask_bbox, -1] = 1

            if img_type == 'outlines' or seg_type == 'outlines':
                outlines = self.get_mask_boundary(cell_mask_bbox)
                if img_type == 'outlines':
                    img_cell_mask[~outlines] = 0
                if seg_type == 'outlines':
                    seg_cell_mask[~outlines] = 0

            # Add the cell mask to the existing overlay
            overlays[0][xmin:xmax, ymin:ymax][cell_mask_bbox] = img_cell_mask[cell_mask_bbox]
            overlays[1][xmin:xmax, ymin:ymax][cell_mask_bbox] = seg_cell_mask[cell_mask_bbox]

        # Update the overlay images
        if drawing_layers:
            layer_overlay[0].setImage(overlays[0])
            layer_overlay[1].setImage(overlays[1])

        # store mask overlays if layer is mask
        if layer == 'mask':
            frame.stored_mask_overlay = overlays

        return

    def get_bounding_box(self, cell_mask):
        # UTIL
        # Find the rows and columns that contain True values
        rows = np.any(cell_mask, axis=1)
        cols = np.any(cell_mask, axis=0)

        # Find the indices of these rows and columns
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]

        # Calculate the bounding box coordinates
        if row_indices.size and col_indices.size:
            min_row, max_row = row_indices[[0, -1]]
            min_col, max_col = col_indices[[0, -1]]
            return min_row, max_row + 1, min_col, max_col + 1
        else:
            # If no True values are found, raise
            return None, None, None, None

    def get_mask_boundary(self, mask):
        # UTIL
        from skimage.segmentation import find_boundaries

        boundaries = find_boundaries(mask, mode='inner')
        return boundaries

    def clear_selection_overlay(self):
        self.selection_overlay[0].clear()
        self.selection_overlay[1].clear()

    def clear_mask_overlay(self):
        self.mask_overlay[0].clear()
        self.mask_overlay[1].clear()

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

    def mouse_moved(self, pos):
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
        if img_data is None:
            img_data = self.img_data
        if seg_data is None:
            seg_data = self.seg_data

        # RGB checkboxes
        self.img_data = img_data.copy()
        self.seg_data = seg_data.copy()

        if RGB_checks is not None:
            for i, check in enumerate(RGB_checks):
                if not check:
                    self.img_data[..., i] = 0

        # update segmentation overlay
        self.draw_outlines()

        # update masks overlay, use the stored overlay if available
        self.overlay_masks()

        # turn seg_data from grayscale to RGBA
        self.seg_data = np.repeat(np.array(self.seg_data[..., np.newaxis]), 4, axis=-1)

        self.img.setImage(self.img_data)
        self.seg.setImage(self.seg_data)

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

        self.setLookupTable('RGB')

        self.img_item = pg.ImageItem(self.image(), levels=(0, 255))
        plot.addItem(self.img_item)
        self.show_grayscale = False
        self.toggle_grayscale()

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
        return rgb_array

    def refresh(self):
        """Refresh the image item."""
        self.img_item.setImage(self.image(), autoLevels=False)

    def setImage(self, img_data):
        self.img_data = img_data
        if img_data.ndim == 2:
            self.red.setImage(self.img_data)
            self.green.clear()
            self.blue.clear()
            self.setLookupTable('gray')
        else:  # RGB image
            if self.img_data.shape[-1] == 2:  # two channels--assume RG and convert to RGB
                self.img_data = np.concatenate((self.img_data, np.zeros((*self.img_data.shape[:-1], 1), dtype=np.uint8)), axis=-1)
            self.red.setImage(self.img_data[..., 0])
            self.green.setImage(self.img_data[..., 1])
            self.blue.setImage(self.img_data[..., 2])
        self.scene.setSceneRect(self.scene.itemsBoundingRect())
        self.refresh()

    def toggle_grayscale(self):
        if self.show_grayscale:
            self.setLookupTable('gray')
        else:
            self.setLookupTable('RGB')

    def setLevels(self, levels):
        """Update the levels of the image items based on the sliders."""
        for level, item in zip(levels, self.channels):
            item.setLevels(level)
        self.refresh()

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
        return [self.create_lut(QColor(255, 0, 0)), self.create_lut(QColor(0, 255, 0)), self.create_lut(QColor(255, 255, 255))]

    def set_grayscale(self, grayscale):
        self.show_grayscale = grayscale
        self.toggle_grayscale()
        self.refresh()


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

    def get_enclosed_pixels(self):
        return self.img_poly.get_enclosed_pixels()


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

    def get_enclosed_pixels(self):
        """Return all pixels enclosed by the polygon."""
        from skimage.draw import polygon

        points = np.array([(p.x(), p.y() - 0.5) for p in self.points])
        points = np.vstack((points, points[0]))  # close the polygon
        line = LineString(points)
        split_line = unary_union(line)
        # Get the bounding box of the polygon
        xmin, ymin = (points.min(axis=0) - 0.5).astype(int)
        all_polygons = list(polygonize(split_line))

        enclosed_pixels = []
        for shapely_polygon in all_polygons:
            poly_verts = np.array(shapely_polygon.exterior.coords)

            # Use skimage.draw.polygon to get the pixels within the polygon
            rr, cc = polygon(poly_verts[:, 1] - ymin, poly_verts[:, 0] - xmin)

            # Adjust the coordinates to the original image space
            polygon_pixels = np.vstack((rr + ymin, cc + xmin)).T
            enclosed_pixels.append(polygon_pixels)

        enclosed_pixels = np.concatenate(enclosed_pixels, axis=0)
        return enclosed_pixels


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
    img = xp.asarray(img)
    if img.ndim == 2:
        inverted = 255 - img
    else:
        hsv = rgb2hsv(img)
        hsv[..., 0] = xp.mod(hsv[..., 0] + 0.5, 1)
        inverted = 1 - hsv2rgb(hsv)

    inverted = (inverted * 255).astype(np.uint8)

    if on_gpu:
        inverted = inverted.get()
    return inverted
