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

    def overlay_outlines(self, event=None, color='white', alpha=0.5):
        ''' Overlay the outlines of the masks on the image plot. '''
        from matplotlib.colors import to_rgb
        
        if not self.parent.outlines_checkbox.isChecked():
            self.img_outline_overlay.clear()
            return
        color=[*to_rgb(color), alpha]

        overlay=np.zeros((*self.parent.frame.masks.shape, 4))
        overlay[self.parent.frame.outlines]=color

        overlay=np.rot90(overlay, 3)
        overlay=np.fliplr(overlay)
        self.img_outline_overlay.setImage(overlay)

    def overlay_masks(self, event=None, alpha=0.5):
        ''' Overlay the masks on both plots. '''
        if not self.parent.masks_checkbox.isChecked():
            # masks are not visible, clear the overlay
            self.mask_overlay[0].clear()
            self.mask_overlay[1].clear()
            return
        
        # get cell colors
        try:
            cell_colors=self.parent.frame.get_cell_attr('color_ID') # retrieve the stored colors for each cell
        except AttributeError:
            #from monolayer_tracking.networks import color_masks, greedy_color # generate pseudo-random colors
            #random_colors=color_masks(self.parent.frame.masks)
            random_colors=np.random.randint(0, self.cell_n_colors, size=self.parent.frame.masks.max())
            cell_colors=self.cell_cmap(random_colors)
            self.parent.frame.set_cell_attr('color_ID', cell_colors)

        # highlight all cells with the specified colors
        cell_indices=np.unique(self.parent.frame.masks)[1:]-1
        img_masks, seg_masks=self.highlight_cells(cell_indices, alpha=alpha, cell_colors=cell_colors, layer='mask')

        self.parent.frame.mask_overlay=[img_masks, seg_masks] # store the overlay for reuse

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
        #if not self.parent.is_grayscale and self.parent.show_grayscale.isChecked():
        #    self.img_data = np.mean(self.img_data, axis=-1) # TODO: incorporate LUTs?

        # update segmentation overlay
        self.overlay_outlines()

        # update masks overlay, use the stored overlay if available
        if hasattr(self.parent.frame, 'mask_overlay') and self.parent.masks_checkbox.isChecked():
            img_masks, seg_masks=self.parent.frame.mask_overlay
            self.mask_overlay[0].setImage(img_masks)
            self.mask_overlay[1].setImage(seg_masks)
        else:
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
        self.redItem=pg.ImageItem(self.img_data[..., 0])
        self.greenItem=pg.ImageItem(self.img_data[..., 1])
        self.blueItem=pg.ImageItem(self.img_data[..., 2])

        # Add items to the view
        plot.addItem(self.redItem)
        plot.addItem(self.greenItem)
        plot.addItem(self.blueItem)

        self.redItem.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        self.greenItem.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        self.blueItem.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        self.RGB_mode()

        self.is_grayscale=False
        self.toggle_grayscale()
    
    def setImage(self, img_data):
        self.img_data=img_data
        self.redItem.setImage(self.img_data[..., 0])
        self.greenItem.setImage(self.img_data[..., 1])
        self.blueItem.setImage(self.img_data[..., 2])

    def toggle_grayscale(self):
        if self.is_grayscale:
            self.grayscale_mode()
        else:
            self.RGB_mode()

    def setLevels(self, levels):
        ''' Update the levels of the image items based on the sliders. '''
        for l, item in zip(levels, [self.redItem, self.greenItem, self.blueItem]):
            item.setLevels(l)

    def create_lut(self, color):
        lut = np.zeros((256, 3), dtype=np.ubyte)
        for i in range(256):
            lut[i] = [color.red() * i // 255, color.green() * i // 255, color.blue() * i // 255]
        return lut
    
    def grayscale_mode(self):
        # Grayscale mode
        gray_lut = self.create_lut(QColor(255, 255, 255))
        self.redItem.setLookupTable(gray_lut)
        self.greenItem.setLookupTable(gray_lut)
        self.blueItem.setLookupTable(gray_lut)
    
    def RGB_mode(self):
        # Color mode
        self.redItem.setLookupTable(self.create_lut(QColor(255, 0, 0)))
        self.greenItem.setLookupTable(self.create_lut(QColor(0, 255, 0)))
        self.blueItem.setLookupTable(self.create_lut(QColor(0, 0, 255)))

    def set_grayscale(self, grayscale):
        self.is_grayscale=grayscale
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
        points=[(p.y()-0.5, p.x()-0.5) for p in self.points]
        shapely_polygon=Polygon(points).buffer(0.1)
        xmin, ymin, xmax, ymax=shapely_polygon.bounds
        x, y = np.meshgrid(np.arange(int(xmin), int(xmax)+1), np.arange(int(ymin), int(ymax)+1))

        bbox_grid=np.vstack((x.flatten(), y.flatten())).T
        enclosed_pixels=np.array([tuple(p) for p in bbox_grid if shapely_polygon.contains(Point(p))])
        
        return enclosed_pixels
