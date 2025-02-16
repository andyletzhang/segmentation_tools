import fastremap
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal


# Common base classes for frame processing
class FrameProcessorSignals(QObject):
    """Base signal class for frame processors"""

    result_ready = pyqtSignal(object, object)  # frame, result


class FrameProcessorTask(QRunnable):
    """Base class for frame processing tasks"""

    def __init__(self, frame):
        super().__init__()
        self.frame = frame
        self.signals = self.__class__.Signals()
        self._is_canceled = False

    def cancel(self):
        self._is_canceled = True

    def check_canceled(self):
        return self._is_canceled


class BoundsCalculatorTask(FrameProcessorTask):
    Signals = FrameProcessorSignals

    def __init__(self, frame, processor):
        super().__init__(frame)
        self.processor = processor

    def run(self):
        if self.check_canceled():
            return

        if hasattr(self.frame, 'bounds'):
            self.signals.result_ready.emit(self.frame, self.frame.bounds)
            return

        try:
            bounds = self.processor.main_window._get_frame_bounds(self.frame)
            if not self.check_canceled():
                self.signals.result_ready.emit(self.frame, bounds)
        except Exception as e:
            print(f'Error calculating bounds: {e}')


class BoundsProcessor:
    """Manages parallel bounds calculation for frames"""

    def __init__(self, parent, n_cores=None):
        self.main_window = parent
        self.thread_pool = QThreadPool.globalInstance()
        if n_cores is not None:
            self.thread_pool.setMaxThreadCount(n_cores)
        self.active_tasks = []

    def process_frames(self, frames):
        """Start bounds calculation for multiple frames"""
        self.abort_all_tasks()
        for frame in frames:
            self.add_frame(frame)

    def add_frame(self, frame):
        """Process bounds for a single frame"""
        task = BoundsCalculatorTask(frame, self)
        task.signals.result_ready.connect(self._handle_bounds_ready)
        self.active_tasks.append(task)
        self.thread_pool.start(task)

    def _handle_bounds_ready(self, frame, bounds):
        """Handle completed bounds calculation"""
        frame.bounds = bounds

    def abort_all_tasks(self):
        """Stop all running tasks"""
        for task in self.active_tasks:
            task.cancel()
        self.thread_pool.clear()
        self.active_tasks.clear()


# Updated mask processing to use common base classes
class MasksLoaderTask(FrameProcessorTask):
    Signals = FrameProcessorSignals

    def __init__(self, frame, processor):
        super().__init__(frame)
        self.processor = processor
        self.alpha = processor.canvas.masks_alpha

    def run(self):
        if self.check_canceled():
            return

        if hasattr(self.frame, 'stored_mask_overlay'):
            self.signals.result_ready.emit(self.frame, self.frame.stored_mask_overlay)
            return

        try:
            canvas = self.processor.canvas
            try:
                cell_colors = self.frame.get_cell_attrs('color_ID')
            except AttributeError:
                cell_colors = canvas.random_cell_color(self.frame.masks.max())
                self.frame.set_cell_attr('color_ID', cell_colors)

            if self.check_canceled():
                return

            cell_indices = fastremap.unique(self.frame.masks)[1:] - 1
            img_masks, seg_masks = canvas.highlight_cells(
                cell_indices, frame=self.frame, alpha=self.alpha, cell_colors=cell_colors, layer='mask'
            )

            if not self.check_canceled():
                self.signals.result_ready.emit(self.frame, [img_masks, seg_masks])
        except Exception as e:
            print(f'Error processing frame: {e}')


class MaskProcessor:
    def __init__(self, canvas, n_cores=None):
        self.canvas = canvas
        self.thread_pool = QThreadPool()
        if n_cores is not None:
            self.thread_pool.setMaxThreadCount(n_cores)
        self.active_tasks = []

    def draw_masks_parallel(self, frames):
        """Processes frames in parallel using QThreadPool."""
        self.abort_all_tasks()
        for frame in frames:
            self.add_frame_task(frame)

    def add_frame_task(self, frame):
        """Processes a single frame in the background."""
        task = MasksLoaderTask(frame, self)
        task.signals.result_ready.connect(self._handle_mask_ready)
        self.active_tasks.append(task)
        self.thread_pool.start(task)

    def _handle_mask_ready(self, frame, overlay):
        """Handle completed mask processing."""
        frame.stored_mask_overlay = overlay

    def abort_all_tasks(self):
        """Stops all running tasks."""
        for task in self.active_tasks:
            task.cancel()
        self.thread_pool.clear()
        self.active_tasks.clear()
