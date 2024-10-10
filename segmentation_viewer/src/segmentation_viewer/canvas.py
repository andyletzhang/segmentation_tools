import numpy as np

import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QGraphicsPolygonItem
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPen, QColor, QBrush, QPolygonF, QPainter
from shapely.geometry import Polygon, Point
    
class PyQtGraphCanvas(QWidget):
    def __init__(self, parent=None, cell_n_colors=10, cell_cmap='tab10'):
        from matplotlib import colormaps
        super().__init__(parent)
        self.parent = parent

        # cell mask colors
        self.cell_n_colors=cell_n_colors
        self.cell_cmap=colormaps.get_cmap(cell_cmap)
        
        # Create a layout for the widget
        plot_layout = QHBoxLayout(self)
        plot_layout.setSpacing(5)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        # Create a PlotWidget for the image and segmentation views
        self.img_plot = SegPlot(title="Image", border="w")
        self.seg_plot = SegPlot(title="Segmentation", border="w")

        plot_layout.addWidget(self.img_plot)
        plot_layout.addWidget(self.seg_plot)

        # Initialize data - 512x512 with white outline
        self.img_data = np.ones((512, 512, 3), dtype=np.uint8)
        self.img_data[5:-5, 5:-5] = 0 # white border
        self.seg_data = self.img_data.copy()

        # Plot the data
        self.img = RGB_ImageItem(self.img_data, parent=self, plot=self.img_plot)
        self.seg = pg.ImageItem(self.seg_data)
        self.img_outline_overlay=pg.ImageItem()
        self.mask_overlay=[pg.ImageItem(), pg.ImageItem()]
        self.selection_overlay=[pg.ImageItem(), pg.ImageItem()]
        self.FUCCI_overlay=[pg.ImageItem(), pg.ImageItem()]
        self.tracking_overlay=[pg.ImageItem(), pg.ImageItem()]

        # add images to the plots
        #self.img_plot.addItem(self.img)
        self.img_plot.addItem(self.img_outline_overlay)

        self.img_plot.addItem(self.mask_overlay[0])
        self.seg_plot.addItem(self.mask_overlay[1])
        
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

    def overlay_outlines(self):
        self.img_outline_overlay.setVisible(self.parent.outlines_checkbox.isChecked())

    def draw_outlines(self, color='white', alpha=0.5):
        ''' Overlay the outlines of the masks on the image plot. '''
        from matplotlib.colors import to_rgb
        color=[*to_rgb(color), alpha]

        overlay=np.zeros((*self.parent.frame.masks.shape, 4))
        overlay[self.parent.frame.outlines]=color

        overlay=np.rot90(overlay, 3)
        overlay=np.fliplr(overlay)
        self.img_outline_overlay.setImage(overlay)

    def overlay_masks(self):
        for l in self.mask_overlay:
            l.setVisible(self.parent.masks_checkbox.isChecked())

        if not hasattr(self.parent.frame, 'stored_mask_overlay'):
            self.draw_masks()
        else:
            for l, overlay in zip(self.mask_overlay, self.parent.frame.stored_mask_overlay):
                l.setImage(overlay)

    def random_cell_color(self, n=1):
        from matplotlib.colors import to_rgb
        random_colors=self.cell_cmap(self.random_color_ID(n))

        return [to_rgb(c) for c in random_colors]
        
    def random_color_ID(self, n=1):
        random_IDs=np.random.randint(0, self.cell_n_colors, size=n)

        if n==1:
            return random_IDs[0]
        else:
            return random_IDs

    def draw_masks(self, alpha=0.5):
        # get cell colors
        try:
            cell_colors=self.parent.frame.get_cell_attrs('color_ID') # retrieve the stored colors for each cell
        except AttributeError:
            #from monolayer_tracking.networks import color_masks, greedy_color # generate pseudo-random colors
            #random_colors=color_masks(self.parent.frame.masks)
            cell_colors=self.random_cell_color(self.parent.frame.masks.max())
            self.parent.frame.set_cell_attr('color_ID', cell_colors)

        # highlight all cells with the specified colors
        cell_indices=np.unique(self.parent.frame.masks)[1:]-1
        img_masks, seg_masks=self.highlight_cells(cell_indices, alpha=alpha, cell_colors=cell_colors, layer='mask')

        return img_masks, seg_masks

    def clear_FUCCI_overlay(self):
        self.FUCCI_overlay[0].clear()
        self.FUCCI_overlay[1].clear()
        
    def clear_tracking_overlay(self):
        self.tracking_overlay[0].clear()
        self.tracking_overlay[1].clear()

    def highlight_cells(self, cell_indices, layer='selection', alpha=0.3, color='white', cell_colors=None, img_type='masks', seg_type='masks'):
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
        
        opaque_mask=mask_overlay.copy()
        opaque_mask[mask_overlay[...,-1]!=0, -1]=1
        if img_type=='outlines':
            mask_overlay[self.parent.frame.outlines==0]=0
        if seg_type=='outlines':
            opaque_mask[self.parent.frame.outlines==0]=0

        mask_overlay=np.rot90(mask_overlay, 3)
        mask_overlay=np.fliplr(mask_overlay)
        opaque_mask=np.rot90(opaque_mask, 3)
        opaque_mask=np.fliplr(opaque_mask)
        
        layer[0].setImage(mask_overlay)
        layer[1].setImage(opaque_mask)

        # store mask overlays if layer is mask
        if layer==self.mask_overlay:
            self.parent.frame.stored_mask_overlay=[mask_overlay, opaque_mask]

        return mask_overlay, opaque_mask

    def add_cell_highlight(self, cell_index, layer='selection', alpha=0.3, color='white', img_type='masks', seg_type='masks'):
        from matplotlib.colors import to_rgb
        masks = self.parent.frame.masks

        # Get the specified overlay layer: selection for highlighting, mask for colored masks
        layer = getattr(self, f'{layer}_overlay')
        
        cell_mask=masks == cell_index + 1
        cell_mask=np.rot90(cell_mask, 3)
        cell_mask=np.fliplr(cell_mask)

        for l in layer:
            if l.image is None: # initialize the overlay
                l.image=np.zeros(cell_mask.shape+(4,))

        if isinstance(color, int):
            color=self.cell_cmap(color)

        elif isinstance(color, str) and color=='none':
            # remove the cell from the overlay
            layer[0].image[cell_mask] = 0
            layer[1].image[cell_mask] = 0

            layer[0].setImage(layer[0].image)
            layer[1].setImage(layer[1].image)
            return None, None
        
        # Convert color to RGBA
        color = [*to_rgb(color), alpha]
        
        # get bounding box of cell mask
        xmin, xmax, ymin, ymax=self.get_bounding_box(cell_mask)

        cell_mask_bbox=cell_mask[xmin:xmax, ymin:ymax]
        # Get the mask for the specified cell
        img_cell_mask = cell_mask_bbox[..., np.newaxis] * color
        # Create an opaque mask for the specified cell
        seg_cell_mask = img_cell_mask.copy()
        seg_cell_mask[cell_mask_bbox, -1] = 1

        if img_type == 'outlines' or seg_type == 'outlines':
            outlines=self.get_mask_boundary(cell_mask_bbox)
            if img_type == 'outlines':
                img_cell_mask[~outlines] = 0
            if seg_type == 'outlines':
                seg_cell_mask[~outlines] = 0

        # Rotate and flip the masks to match the display orientation
        # Add the cell mask to the existing overlay
        layer[0].image[xmin:xmax, ymin:ymax][cell_mask_bbox] = img_cell_mask[cell_mask_bbox]
        layer[1].image[xmin:xmax, ymin:ymax][cell_mask_bbox] = seg_cell_mask[cell_mask_bbox]

        # Update the overlay images
        layer[0].setImage(layer[0].image)
        layer[1].setImage(layer[1].image)

        # store mask overlays if layer is mask
        if layer==self.mask_overlay:
            self.parent.frame.stored_mask_overlay=[layer[0].image, layer[1].image]

        return img_cell_mask, seg_cell_mask
    
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
            return min_row, max_row+1, min_col, max_col+1
        else:
            # If no True values are found, return None
            return None

    def get_mask_boundary(self, mask):
        # UTIL
        from skimage.segmentation import find_boundaries
        boundaries=find_boundaries(mask, mode='inner')
        return boundaries
    
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

    def update_display(self, img_data=None, seg_data=None, RGB_checks=None):
        """Update the display when checkboxes change."""
        if img_data is not None:
            self.img_data = np.rot90(img_data, 3).copy()
            # invert y axis
            self.img_data = np.fliplr(self.img_data)
        if seg_data is not None:
            self.seg_data = np.rot90(seg_data, 3).copy()
            # invert y axis
            self.seg_data = np.fliplr(self.seg_data)

        # RGB checkboxes
        if RGB_checks is not None:
            for i, check in enumerate(RGB_checks):
                if not check:
                    self.img_data[..., i] = 0
            
        # Grayscale checkbox
        #if not self.parent.show_grayscale and self.parent.show_grayscale.isChecked():
        #    self.img_data = np.mean(self.img_data, axis=-1) # TODO: incorporate LUTs?

        # update segmentation overlay
        self.draw_outlines()

        # update masks overlay, use the stored overlay if available
        self.overlay_masks()

        # turn seg_data from grayscale to RGBA
        self.seg_data=np.repeat(np.array(self.seg_data[..., np.newaxis]), 4, axis=-1)

        self.img.setImage(self.img_data)
        self.seg.setImage(self.seg_data)

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
    '''
    Custom PlotWidget for the segmentation viewer. 
    Only functional difference is to redirect mouse wheel events to the main window.
    '''
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.setMenuEnabled(False)
        self.getViewBox().invertY(True)
        self.setAspectLocked(True)
        self.setContentsMargins(0, 0, 0, 0)

    def wheelEvent(self, event):
        # ctrl+wheel zooms like normal, otherwise send it up to the main window
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(event)
        else:
            event.ignore()

class RGB_ImageItem():
    def __init__(self, img_data=None, parent=None, plot:pg.PlotWidget=None):
        self.parent=parent
        if img_data is None:
            img_data=np.zeros((512, 512, 3), dtype=np.uint8)
        self.img_data=img_data
        self.red=pg.ImageItem(self.img_data[..., 0])
        self.green=pg.ImageItem(self.img_data[..., 1])
        self.blue=pg.ImageItem(self.img_data[..., 2])

        # Add items to the view
        plot.addItem(self.red)
        plot.addItem(self.green)
        plot.addItem(self.blue)

        self.red.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        self.green.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        self.blue.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        self.setLookupTable('RGB')

        self.show_grayscale=False
        self.is_grayscale=False
        self.toggle_grayscale()
    
    def setImage(self, img_data):
        self.img_data=img_data
        if img_data.ndim==2:
            self.is_grayscale=True
            self.red.setImage(self.img_data)
            self.green.clear()
            self.blue.clear()
            self.setLookupTable('gray')
        else: # RGB image
            self.red.setImage(self.img_data[..., 0])
            self.green.setImage(self.img_data[..., 1])
            self.blue.setImage(self.img_data[..., 2])
            self.setLookupTable('RGB')

    def toggle_grayscale(self):
        if self.show_grayscale:
            self.setLookupTable('gray')
        else:
            self.setLookupTable('RGB')

    def setLevels(self, levels):
        ''' Update the levels of the image items based on the sliders. '''
        for l, item in zip(levels, [self.red, self.green, self.blue]):
            item.setLevels(l)

    def create_lut(self, color):
        lut = np.zeros((256, 3), dtype=np.ubyte)
        for i in range(256):
            lut[i] = [color.red() * i // 255, color.green() * i // 255, color.blue() * i // 255]
        return lut
    
    def setLookupTable(self, lut_style):
        ''' Set the lookup table for the image items. '''
        if lut_style=='gray':
            luts=self.gray_lut()
        elif lut_style=='RGB':
            luts=self.RGB_lut()

        for item, lut in zip([self.red, self.green, self.blue], luts):
            item.setLookupTable(lut)

    def gray_lut(self):
        return [self.create_lut(QColor(255, 255, 255))]*3

    def RGB_lut(self):
        return [self.create_lut(QColor(255, 0, 0)), self.create_lut(QColor(0, 255, 0)), self.create_lut(QColor(0, 0, 255))]

    def set_grayscale(self, grayscale):
        self.show_grayscale=grayscale
        self.toggle_grayscale()

class CellMaskPolygon(QGraphicsPolygonItem):
    ''' Polygonal overlay for drawing the current mask. '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setPen(QPen(QColor(255, 255, 255, 50)))
        self.setBrush(QBrush(QColor(120, 120, 120, 50)))
        self.points=[]

    def clearPoints(self):
        self.points=[]
        self.update_polygon()

    def update_polygon(self):
        polygon=QPolygonF(self.points)
        self.setPolygon(polygon)

    def add_vertex(self, y, x):
        y, x = y+0.5, x+0.5
        self.points.append(QPointF(y, x))
        self.update_polygon()
        self.last_handle_pos = (y, x)

    def get_enclosed_pixels(self):
        ''' Return all pixels enclosed by the polygon. '''
        from skimage.draw import polygon

        points = [(p.x(), p.y()-0.5) for p in self.points]
        shapely_polygon = Polygon(points).buffer(0.1)
        
        # Get the bounding box of the polygon
        xmin, ymin, xmax, ymax = shapely_polygon.bounds
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        
        # Get the coordinates of the polygon vertices
        poly_verts = np.array(shapely_polygon.exterior.coords)
        
        # Use skimage.draw.polygon to get the pixels within the polygon
        rr, cc = polygon(poly_verts[:, 1] - ymin, poly_verts[:, 0] - xmin)
                
        # Adjust the coordinates to the original image space
        enclosed_pixels = np.vstack((rr + ymin, cc + xmin)).T
        
        return enclosed_pixels
