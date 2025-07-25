from glob import glob
from pathlib import Path

import fastremap
import numpy as np
import pandas as pd
from natsort import natsorted
from scipy import ndimage
import copy

from . import preprocessing
from .utils import masks_to_outlines, outlines_list

from typing import Union, Optional

"""
    Stacks are collections of time lapse images on a single stage.
    Images are individual seg.npy files, presumed to be segmented membranes generated with cellpose.
    Cells are generated for each cell mask identified by segmentation.
"""

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


class Cell:
    """Class for each labeled cell membrane."""

    def __init__(self, n, outline=None, parent=None, **kwargs):
        if parent is None:
            raise ValueError('Cell must be initialized with a parent SegmentedImage object.')
        self.n = n
        if outline is not None:
            self._outline = outline
        self.parent = parent

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def frame(self):
        """Get frame number from parent or return None."""
        try:
            return self.parent.frame_number
        except AttributeError:
            return None

    @property
    def outline(self):
        if not hasattr(self, '_outline'):
            self._outline = outlines_list(self.mask)[0]
        return self._outline

    @outline.setter
    def outline(self, outline):
        self._outline = outline
        
    @property
    def scale(self):
        """Get scale from parent or return None."""
        try:
            return self.parent.scale
        except AttributeError:
            return None

    @property
    def area_pixels(self):
        area = 0.5 * np.abs(
            np.dot(self.outline.T[0], np.roll(self.outline.T[1], 1)) - 
            np.dot(self.outline.T[1], np.roll(self.outline.T[0], 1))
        )
        return area
        
    @property
    def area(self):
        """Area in scaled units."""
        pixels = self.area_pixels
        if pixels is None or self.scale is None:
            return None
        return pixels * self.scale**2
    
    @property
    def perimeter_pixels(self):
        if len(self.outline) == 0:
            return 0
        else:
            perimeter = np.sum(np.linalg.norm(np.diff(self.outline, axis=0, append=[self.outline[0]]).T, axis=0))
            return perimeter
            
    @property
    def perimeter(self):
        """Perimeter in scaled units."""
        pixels = self.perimeter_pixels
        if pixels is None or self.scale is None:
            return None
        return pixels * self.scale

    @property
    def circularity(self):
        circularity = 4 * np.pi * self.area_pixels / self.perimeter_pixels**2
        return circularity

    @property
    def centroid(self):
        if hasattr(self, '_centroid'):
            return self._centroid
        else:  # centroid via Green's theorem
            self.get_centroid()
            return self._centroid

    @property
    def corrected_centroid(self):
        if not hasattr(self.parent, 'drift'):
            return self.centroid
        else:
            return self.centroid - self.parent.drift

    @centroid.setter
    def centroid(self, centroid):
        self._centroid = centroid

    def get_centroid(self):
        x = self.outline[:, 0]
        y = self.outline[:, 1]
        A = self.area_pixels
        Cx = np.sum((x + np.roll(x, 1)) * (x * np.roll(y, 1) - np.roll(x, 1) * y))
        Cy = np.sum((y + np.roll(y, 1)) * (x * np.roll(y, 1) - np.roll(x, 1) * y))

        self._centroid = np.array([Cx, Cy]) / (6 * A)

        return self._centroid

    @property
    def mask(self):
        return self.parent.masks == self.n + 1

    def sort_vertices(self):
        """
        determines which vertices are connected by ordering polar angles to each vertex w.r.t. the centroid.
        some edge cases where this won't work (unusual concave structures, radial line segments) but I think these are sufficiently unlikely in physiological cells.
        could replace this with a sort_vertices which pulls them by outline now that I have that.
        only calculable for cells with reconstructed vertices ('good cells').
        """
        
        zeroed_coords = self.vertices - self.centroid[::-1]  # zero vertices to the centroid
        angles = np.arctan2(zeroed_coords[:, 0], zeroed_coords[:, 1])  # get polar angles
        vertex_order = np.argsort(angles)  # sort polar angles
        return self.vertices[vertex_order]

    @property
    def sorted_vertices(self):
        if hasattr(self, '_sorted_vertices'):
            return self._sorted_vertices
        else:
            self._sorted_vertices = self.sort_vertices()
            return self._sorted_vertices

    @sorted_vertices.setter
    def sorted_vertices(self, vertices):
        self._sorted_vertices = vertices

    def TCJ_axis(self):
        """
        Compute the axis orientation based on the spatial distribution of tricellular junctions (TCJs).

        Returns:
            float or bool: Orientation angle of the axis if computation is successful, False otherwise.
        """

        try:
            TCJs = np.flip(self.sorted_vertices, axis=1)
        except AttributeError:
            # If the cell doesn't have a full set of neighbors or get_TCJs() hasn't been run
            return False

        # Calculate the center of mass of TCJs
        CoM_J = TCJs.mean(axis=0)

        # Calculate the scatter matrix
        a = [np.outer(junction - CoM_J, junction - CoM_J) for junction in TCJs]
        S = np.mean(a, axis=0)

        # Compute eigenvalues and eigenvectors of the scatter matrix
        eigenvalues, eigenvectors = np.linalg.eig(S)

        # Determine the major axis and calculate its angle
        shapeAxis = eigenvectors.T[np.argmax(eigenvalues)]
        self.circ_J = np.min([eigenvalues[0] / eigenvalues[1], eigenvalues[1] / eigenvalues[0]])
        self.theta_J = np.arctan2(*np.flip(shapeAxis))

        return self.theta_J

    def perimeter_axis(self):
        """
        Compute the axis orientation based on the perimeter of the cell.

        Returns:
            float: Orientation angle of the axis.
        """
        vertices = self.outline

        # Calculate segment lengths
        segment_lengths = np.linalg.norm(np.diff(vertices, append=[vertices[0]], axis=0), axis=1)

        # Calculate center of mass of the perimeter
        CoM_P = (
            np.sum(
                [1 / 2 * (vertices[i] + vertices[(i + 1) % len(vertices)]) * segment_lengths[i] for i in range(len(vertices))],
                axis=0,
            )
            / self.perimeter_pixels
        )

        # Translate vertices to center of mass
        zeroed_vertices = vertices - CoM_P

        # Calculate scatter matrix
        summation = []
        for i in range(len(vertices)):
            v_current = zeroed_vertices[i]
            v_next = zeroed_vertices[(i + 1) % len(vertices)]
            summation.append(
                segment_lengths[i]
                * (
                    (np.outer(v_next, v_next) + np.outer(v_current, v_current)) / 3
                    + (np.outer(v_next, v_current) + np.outer(v_current, v_next)) / 6
                )
            )
        S_P = np.sum(summation, axis=0)

        # Compute eigenvalues and eigenvectors of the scatter matrix
        eigenvalues, eigenvectors = np.linalg.eig(S_P)

        # Determine the major axis and calculate its angle
        shapeAxis = eigenvectors.T[np.argmax(eigenvalues)]
        self.circ_P = np.min([eigenvalues[0] / eigenvalues[1], eigenvalues[1] / eigenvalues[0]])
        self.theta_P = np.arctan2(*np.flip(shapeAxis))

        return self.theta_P

    def poly_perimeter_axis(self):
        """
        Compute the axis orientation based on the perimeter of the polygon formed by connecting TCJs.

        Returns:
            float or bool: Orientation angle of the axis if computation is successful, False otherwise.
        """
        try:
            TCJs = np.flip(self.sorted_vertices, axis=1)
        except AttributeError:
            # If the cell doesn't have a full set of neighbors or get_TCJs() hasn't been run
            return False

        # Calculate segment lengths of the polygon
        poly_segment_lengths = np.linalg.norm(np.diff(TCJs, append=[TCJs[0]], axis=0), axis=1)
        poly_perimeter = poly_segment_lengths.sum()

        # Calculate center of mass of the polygon perimeter
        CoM_P = (
            np.sum([1 / 2 * (TCJs[i] + TCJs[(i + 1) % len(TCJs)]) * poly_segment_lengths[i] for i in range(len(TCJs))], axis=0)
            / poly_perimeter
        )

        # Translate TCJs to center of mass
        zeroed_TCJs = TCJs - CoM_P

        # Calculate scatter matrix
        summation = []
        for i in range(len(TCJs)):
            v_current = zeroed_TCJs[i]
            v_next = zeroed_TCJs[(i + 1) % len(TCJs)]
            summation.append(
                poly_segment_lengths[i]
                * (
                    (np.outer(v_next, v_next) + np.outer(v_current, v_current)) / 3
                    + (np.outer(v_next, v_current) + np.outer(v_current, v_next)) / 6
                )
            )
        S_P = np.sum(summation, axis=0)

        # Compute eigenvalues and eigenvectors of the scatter matrix
        eigenvalues, eigenvectors = np.linalg.eig(S_P)

        # Determine the major axis and calculate its angle
        shapeAxis = eigenvectors.T[np.argmax(eigenvalues)]
        self.circ_polyP = np.min([eigenvalues[0] / eigenvalues[1], eigenvalues[1] / eigenvalues[0]])
        self.theta_polyP = np.arctan2(*np.flip(shapeAxis))

        return self.theta_polyP

    @property
    def shape_parameter(self):
        """
        calculates the shape parameter q=perimeter/area**2 using vertex data.
        only calculable for cells with reconstructed vertices ('good cells').
        """
        shape_parameter = self.vertex_perimeter / (self.vertex_area**0.5)
        return shape_parameter

    @property
    def vertex_area(self):
        """returns the area of the cell as calculated from the vertices."""
        y, x = self.sorted_vertices.T
        vertex_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return vertex_area

    @property
    def vertex_perimeter(self):
        """returns the perimeter of the cell as calculated from the vertices."""
        vertex_perimeter = np.sum(np.linalg.norm(np.diff(self.sorted_vertices, append=[self.sorted_vertices[0]], axis=0), axis=1))
        return vertex_perimeter

    def fit_ellipse(self):
        """
        uses skimage's EllipseModel to fit an ellipse to the outline of the cell.
        """
        from skimage.measure import EllipseModel

        ellipse_model = EllipseModel()

        ellipse_model.estimate(self.outline)
        if ellipse_model.params:
            a, b, theta = ellipse_model.params[2:]
        else:
            return False

        self.fit_params = (
            ellipse_model.params
        )  # mildly redundant but preserve all the fit parameters for more convenient plotting

        if a > b:
            self.aspect = a / b
            self.theta = theta % np.pi
        else:
            self.aspect = b / a
            self.theta = (theta + np.pi / 2) % np.pi

        return self.fit_params

    def copy(self):
        return copy.copy(self)


class SegmentedStack:
    """
    A base class for time lapse or multipoint data.
    Built around a collection of SegmentedImage objects, stored in the frames attribute.
    """

    def __init__(
        self,
        from_frames=[],
        stack_path=None,
        frame_paths=None,
        coarse_grain=1,
        verbose_load=False,
        progress_bar=lambda x: x,
        stack_type='time',
        **kwargs,
    ):
        """
        Stacks can be initialized from:
            1. a list of already initialized SegmentedImage objects (from_frames).
            2. a directory of seg.npy files (stack_path),
            3. a list of seg.npy file paths (frame_paths).

        Args:
            from_frames (list): List of SegmentedImage objects to initialize the stack with.
            stack_path (str): Path to a directory containing seg.npy files.
            frame_paths (list): List of paths to seg.npy files.
            coarse_grain (int): Number of frames to skip when loading a stack from a directory.
            verbose_load (bool): If True, prints the number of seg.npy files found and loaded.
            progress_bar (function): Function to display a progress bar.
            stack_type (str): Type of stack to initialize. Options are 'time' or 'multipoint'.
        """
        self.progress_bar = progress_bar
        if len(from_frames) > 0:
            self.frames = np.array(from_frames)
            self.name = str(Path(self.frames[0].name).parent) + '/'
            for n, frame in enumerate(self.frames):
                frame.frame_number = n

        elif stack_path:
            self.frame_paths = natsorted(glob(stack_path + '/*seg.npy'))[
                0::coarse_grain
            ]  # find and alphanumerically sort all segmented images at the given path
            if len(self.frame_paths) == 0:
                raise FileNotFoundError(
                    'No seg.npy files found at {}'.format(stack_path)
                )  # warn me if I can't find anything (probably a formatting error)
            if verbose_load:
                print(f'{len(self.frame_paths)} segmented files found at {stack_path}, loading...')
            self.frames = np.array(
                [
                    SegmentedImage(frame_path, frame_number=n, **kwargs)
                    for n, frame_path in enumerate(self.progress_bar(self.frame_paths))
                ]
            )  # load all segmented images
            if not stack_path.endswith('/') or not stack_path.endswith('\\'):
                stack_path += '/'
            self.name = stack_path

        elif frame_paths:
            self.frames = np.array(
                [
                    SegmentedImage(frame_path, frame_number=n, **kwargs)
                    for n, frame_path in enumerate(self.progress_bar(frame_paths))
                ]
            )  # load all segmented images
            self.name = str(Path(frame_paths[0]).parent) + '/'

        else:
            raise ValueError('Either from_frames, stack_path or frame_paths must be provided to load a Stack.')

        if stack_type == 'time':
            self.__class__ = TimeStack
        elif stack_type == 'multipoint':
            pass
        else:
            raise ValueError('Invalid stack type. Recognized formats are "time" or "multipoint".')

    # -------------Indexing Lower Objects-------------
    def densities(self):
        """returns an array of densities for each frame in chronological order"""
        return np.array([frame.mean_density() for frame in self.frames])

    def centroids(self):
        """returns a DataFrame of cell centroids, with additional columns specifying the cell and frame numbers. Used for particle tracking."""
        all_centroids = []
        for frame_number, frame in enumerate(self.frames):  # iterate through frames
            centroids = pd.DataFrame(frame.centroids(), columns=['y', 'x'])  # get centroids
            centroids['frame'] = frame_number  # label each set of centroids with their frame number
            all_centroids.append(centroids)
        all_centroids = pd.concat(all_centroids).rename_axis('cell_number').reset_index()
        return all_centroids

    def shape_parameters(self):
        """returns a DataFrame of shape parameters, with additional columns specifying the cell and frame numbers. Only returns cells with a full set of neighbors"""
        all_q = []
        for frame_number, frame in enumerate(self.frames):  # iterate through frames
            q = pd.DataFrame(frame.shape_parameters(), columns=['q'])  # get shape parameter
            q['frame'] = frame_number  # label each set of shape parameters with their frame number
            all_q.append(q)
        all_q = pd.concat(all_q).rename_axis('cell_number').reset_index()
        return all_q

    # -------------I/O Image/Segmentation-------------
    def load_img(self, **kwargs):
        for frame in self.frames:
            frame.load_img(**kwargs)

    def delete_frame(self, frame_number):
        self.frames = np.delete(self.frames, frame_number)
        self.renumber_frames()

        if hasattr(self, 'tracked_centroids'):
            self.tracked_centroids = self.tracked_centroids[self.tracked_centroids['frame'] != frame_number]
            self.tracked_centroids.loc[self.tracked_centroids['frame'] > frame_number, 'frame'] -= 1

    def make_substack(self, frame_numbers: np.ndarray):
        self.frames = self.frames[frame_numbers]
        self.renumber_frames()

        if hasattr(self, 'tracked_centroids'):
            self.tracked_centroids = self.tracked_centroids[self.tracked_centroids['frame'].isin(frame_numbers)]
            self.tracked_centroids['frame'] = self.tracked_centroids['frame'].map(
                {frame_number: n for n, frame_number in enumerate(frame_numbers)}
            )

    def renumber_frames(self):
        for n, frame in enumerate(self.frames):
            frame.frame_number = n

    # -------------I/O Tracking Data---------------
    def load_tracking(self, tracking_path=None):
        if tracking_path is None:
            tracking_path = Path(self.name).with_name('tracking.csv')
        if not Path(tracking_path).exists():
            raise FileNotFoundError(f'No tracking data found at {tracking_path}.')
        self.tracked_centroids = pd.read_csv(
            tracking_path,
            usecols=['cell_number', 'y', 'x', 'frame', 'particle'],
            dtype={'frame': int, 'particle': int, 'cell_number': int},
            index_col=False,
        )
        self.fix_tracked_centroids()
        self.__class__ = TimeStack

    def fix_tracked_centroids(self, t=None, inplace=True):
        """make sure every cell is accounted for in the tracking data."""
        if t is None:
            t = self.tracked_centroids
        for frame in self.frames:
            tracked_frame = t[t.frame == frame.frame_number]
            tracked_cells = tracked_frame['cell_number']
            frame_cells = frame.get_cell_attrs('n')

            missing_tracks = set(frame_cells) - set(tracked_cells)
            if len(missing_tracks) > 0:
                print(f'tracked_centroids is missing {len(missing_tracks)} cells in frame {frame.frame_number}: {missing_tracks}')
                new_particle_numbers = np.arange(len(missing_tracks)) + t['particle'].max() + 1
                new_particles = pd.DataFrame(
                    [
                        [cell.n, cell.centroid[0], cell.centroid[1], frame.frame_number, particle_number]
                        for cell, particle_number in zip(frame.cells[list(missing_tracks)], new_particle_numbers)
                    ],
                    columns=['cell_number', 'y', 'x', 'frame', 'particle'],
                )
                t = pd.concat([t, new_particles])

            extra_tracks = set(tracked_cells) - set(frame_cells)
            if len(extra_tracks) > 0:
                print(f'tracked_centroids has {len(extra_tracks)} extra tracks in frame {frame.frame_number}: {extra_tracks}')
                t.drop(tracked_frame[tracked_frame.cell_number.isin(extra_tracks)].index, inplace=True)

        t = t.sort_values(['frame', 'particle'])
        t['particle'] = t.groupby('particle').ngroup()  # renumber particles contiguously

        if inplace:
            self.tracked_centroids = t
        return t

    def save_tracking(self, file_path=None):
        if file_path is None:
            file_path = Path(self.name).with_name('tracking.csv')
        self.tracked_centroids[['cell_number', 'y', 'x', 'frame', 'particle']].to_csv(file_path, index=False)

    # -------------Mask Operations-------------
    def rot90(self, k=1, progress=lambda x: x):
        for frame in progress(self.frames):
            frame.rot90(k)

    def delete_cells(self, cell_numbers, frame_number):
        frame = self.frames[frame_number]
        idx = frame.delete_cells(cell_numbers)
        return idx

    def remove_edge_cells(self, frames=None, margin=1):
        if frames is None:
            frames = self.frames
        all_edge_cells = []
        for frame in frames:
            edge_cells = frame.find_edge_cells(margin)
            all_edge_cells.append(edge_cells)
            if len(edge_cells) > 0:
                self.delete_cells(edge_cells, frame.frame_number)
                frame.outlines = masks_to_outlines(frame.masks)
        return all_edge_cells

    def merge_cells(self, cell_n1, cell_n2, frame_number):
        frame = self.frames[frame_number]
        new_cell, idx = frame.merge_cells(cell_n1, cell_n2)
        return new_cell, idx

    def add_cell(self, cell: Cell, mask: np.ndarray):
        frame = self.frames[cell.frame]
        frame.add_cell(cell, mask=mask)

    def remove_cell(self, cell, mask: Optional[np.ndarray] = None):
        frame = self.frames[cell.frame]
        frame.remove_cell(cell, mask=mask)

    # -------------Magic-------------
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]


class TimeStack(SegmentedStack):
    """Time lapse data. Provides methods for tracking cells through time, identifying cell cycle and mitotic events."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # -------------Particle Tracking-------------
    def track_centroids(self, memory=0, v_quantile=0.97, filter_stubs=False, search_range=None, **kwargs):
        """
        uses the trackpy package to track cell centroids.
        This works in two stages:
            1. a first tracking with a large search range which identifies and corrects drift and determines a characteristic velocity, and
            2. a second, more discerning tracking with a search range based on the maximum cell speed.
        **kwargs go to tp.link(): 'search_range' depends on cell diameter, and 'memory' can account for the disappearance of a centroid for a certain number of frames (default 0).
        """
        import trackpy as tp

        tp.quiet()
        # preliminary search radius for the first pass at tracking. This doesn't have to be perfect, just in the right ballpark (too high better than too low).
        # shifts in the FOV which I've observed are up to ~16 pixels large. Any tracking that wants to accomodate for these must be >16.
        max_search_range = int(
            np.nanmean([np.sqrt(np.quantile(frame.cell_areas(scaled=False), 0.9)) for frame in self.frames]) * 2 / 3
        )  # 2/3 a large cell: overshoot the largest tracking range by a bit.

        # FIRST PASS: large search range
        t_firstpass = tp.link(
            self.centroids(), search_range=max_search_range, adaptive_step=0.96, adaptive_stop=2
        )  # use adaptive search range to find the most tracks without throwing an error

        drift_firstpass = tp.compute_drift(t_firstpass)
        t_corrected = tp.subtract_drift(t_firstpass, drift_firstpass).reset_index(
            drop=True
        )  # subtract drift in the stage. NB: will also catch any collective migration.

        if search_range is None:  # get a search range by evaluating the velocity distribution.
            # This is a simplified copy of the get_velocities() method.
            v = t_corrected[['x', 'y', 'frame', 'particle']].groupby('particle').apply(np.diff, axis=0)
            v_arr = np.concatenate(v.values)[:, :3]
            velocities = np.linalg.norm(v_arr[:, :2], axis=1) / v_arr[:, 2]  # get magnitude of velocity normalized by time

            search_range = np.quantile(
                velocities, v_quantile
            )  # let's assume the top (1-v_quantile) percent of velocities are spurious tracks or actually mitosis. We'll use this as a cutoff for the real tracking.

        # FINAL PASS: small search range
        t_final = tp.link(t_corrected, search_range=search_range, memory=memory, **kwargs)
        if filter_stubs:
            t_final = tp.filter_stubs(t_final, filter_stubs)  # drop tracks which are $filter_stubs or fewer frames long
            t_final['particle'] = t_final.groupby(
                'particle'
            ).ngroup()  # renumber particle tracks with filtered tracks ommitted from count
        drift_finalpass = tp.compute_drift(t_final)
        self.tracked_centroids = tp.subtract_drift(t_final, drift_finalpass).reset_index(drop=True)
        self.drift = drift_firstpass + drift_finalpass
        for frame, drift in zip(self.frames, self.drift.values):
            frame.drift = drift
        self.drift = pd.concat(
            [pd.DataFrame({'frame': [0], 'y': [0], 'x': [0]}).set_index('frame'), self.drift]
        )  # add a row for the first frame (no drift)
        self.tracking_range = search_range
        return self.tracked_centroids

    # -------------Mask Operations-------------
    def rot90(self, k=1, **kwargs):
        from .utils import rotate_points

        shape = self.frames[0].resolution
        super().rot90(k, **kwargs)
        for attr in ['tracked_centroids', 'drift', 'velocities']:
            if hasattr(self, attr):
                values = getattr(self, attr)
                values[['y', 'x']] = rotate_points(values[['y', 'x']].values, shape, k)

    def merge_cells(self, cell_n1, cell_n2, frame_number):
        new_cell, idx = super().merge_cells(cell_n1, cell_n2, frame_number)
        if hasattr(self, 'tracked_centroids'):
            # recompute merged cell centroid
            t = self.tracked_centroids
            t.loc[(t.frame == frame_number) & (t.cell_number == cell_n1), ['x', 'y']] = new_cell.centroid.astype(t['x'].dtype)
            self.remove_tracking_data([cell_n2], idx, frame_number)
        return new_cell, idx

    def delete_cells(self, cell_numbers, frame_number):
        idx = super().delete_cells(cell_numbers, frame_number)
        if hasattr(self, 'tracked_centroids'):
            self.remove_tracking_data(cell_numbers, idx, frame_number)

    def remove_tracking_data(self, cell_numbers, idx, frame_number):
        t = self.tracked_centroids
        t.drop(t[(t.frame == frame_number) & np.isin(t.cell_number, cell_numbers)].index, inplace=True)

        # remap cell numbers in tracked_centroids
        cell_remap = fastremap.component_map(idx, np.arange(len(idx)))
        t.loc[t.frame == frame_number, 'cell_number'] = (
            t.loc[t.frame == frame_number, 'cell_number'].map(cell_remap).astype(t.cell_number.dtype)
        )

    def add_cell(self, cell: Cell, mask: np.ndarray):
        super().add_cell(cell, mask=mask)
        frame_number = cell.frame
        if hasattr(self, 'tracked_centroids'):
            t = self.tracked_centroids
            t.loc[(t.frame == frame_number) & (t.cell_number >= cell.n), 'cell_number'] += 1
            new_particle_ID = t['particle'].max() + 1
            data = {
                'cell_number': cell.n,
                'y': cell.centroid[0],
                'x': cell.centroid[1],
                'frame': frame_number,
                'particle': new_particle_ID,
            }
            if hasattr(cell, 'color_ID'):
                data['color_ID'] = cell.color_ID
            placeholder_particle = {idx: None for idx in t.columns}
            placeholder_particle.update({k: v for k, v in data.items() if k in t.columns})
            self.tracked_centroids = pd.concat([t, pd.DataFrame(placeholder_particle, index=[t.index.max() + 1])])

    def remove_cell(self, cell, mask: Optional[np.ndarray] = None):
        super().remove_cell(cell, mask=mask)
        frame_number = cell.frame
        if hasattr(self, 'tracked_centroids'):
            t = self.tracked_centroids
            t.drop(t[(t.frame == frame_number) & (t.cell_number == cell.n)].index, inplace=True)
            t.loc[(t.frame == frame_number) & (t.cell_number > cell.n), 'cell_number'] -= 1

    # -------------Particle Operations-------------
    def merge_particle_tracks(self, first_ID, second_ID, frame):
        """
        Merge two particle tracks into a single track.

        Args:
            first_ID (int): ID of the first particle to merge.
            second_ID (int): ID of the second particle to merge.
            frame (int): Frame number at which second_ID should be merged into first_ID.

        Returns:
            DataFrame: DataFrame containing the merged particle track.
        """
        t = self.tracked_centroids
        first_particle = t.loc[t.particle == first_ID]
        second_particle = t.loc[t.particle == second_ID]
        # if necessary, split particles at the frame of interest
        new_head, new_tail = None, None
        if np.any([first_particle.frame >= frame]):
            new_tail = self.split_particle_track(first_ID, frame)
        if np.any([second_particle.frame < frame]):
            new_head = second_ID
            second_ID = self.split_particle_track(second_ID, frame)  # reassign second_ID in case of split

        # merge particles
        t['particle'] = t['particle'].replace({second_ID: first_ID})
        if hasattr(self, 'mitoses'):
            self.mitoses[['mother', 'daughter1', 'daughter2']] = self.mitoses[['mother', 'daughter1', 'daughter2']].replace(
                {second_ID: first_ID}
            )

        # new merged ID
        f, c = t.loc[t.particle == first_ID][['frame', 'cell_number']].values[0]

        # renumber particles so that they're contiguous
        t['particle'] = self.renumber_particles()

        merged = t.loc[(t.frame == f) & (t.cell_number == c)]['particle'].iloc[0]
        return merged, new_head, new_tail

    def renumber_particles(self):
        """
        Renumber particle IDs so that they are contiguous.
        """
        t = self.tracked_centroids
        t['particle'], remap = fastremap.renumber(t['particle'].values)

        if hasattr(self, 'mitoses'):
            self.mitoses[['mother', 'daughter1', 'daughter2']] = self.mitoses[['mother', 'daughter1', 'daughter2']].replace(remap)
        return t['particle']

    def split_particle_track(self, particle_ID, split_frame):
        """
        Split a particle track into two separate tracks at a given frame.

        Args:
            particle_ID (int): ID of the particle to split.
            split_frame (int): Frame number at which to split the particle track.
            **kwargs: Additional keyword arguments passed to trackpy.link() function.

        Returns:
            int: ID of the new particle track created by the split.
        """
        t = self.tracked_centroids
        new_particle_ID = t['particle'].max() + 1
        if np.sum((t.particle == particle_ID) & (t.frame < split_frame)) == 0:
            # if the particle doesn't exist before the split frame, nothing to do
            return None
        else:
            t.loc[(t.particle == particle_ID) & (t.frame >= split_frame), 'particle'] = new_particle_ID
            if hasattr(self, 'mitoses'):
                needs_editing = ((self.mitoses[['mother', 'daughter1', 'daughter2']] == particle_ID).any(axis=1)) & (
                    self.mitoses['frame'] >= split_frame
                )
                self.mitoses.loc[needs_editing, ['mother', 'daughter1', 'daughter2']] = self.mitoses.loc[
                    needs_editing, ['mother', 'daughter1', 'daughter2']
                ].replace({particle_ID: new_particle_ID})
            return new_particle_ID

    # -------------Retrieve Tracking Data-------------
    def particle_from_cell(self, cell_number, frame_number):
        if not hasattr(self, 'tracked_centroids'):
            raise ValueError('No tracking data available.')
        t = self.tracked_centroids
        particle = t[(t.frame == frame_number) & (t.cell_number == cell_number)]['particle']
        if len(particle) == 1:
            return particle.item()
        elif len(particle) > 1:
            raise ValueError(f'Cell {cell_number} has multiple particles in frame {frame_number}')
        else:
            return None

    def cell_from_particle(self, particle, frame_number):
        if not hasattr(self, 'tracked_centroids'):
            raise ValueError('No tracking data available.')
        t = self.tracked_centroids
        cell = t[(t.frame == frame_number) & (t.particle == particle)]['cell_number']
        if len(cell) == 1:
            return cell.item()
        elif len(cell) > 1:
            raise ValueError(f'Particle {particle} has {len(cell)} cells in frame {frame_number}')
        else:
            return None

    def get_particle(self, input_ID):
        """
        Retrieve the cell trajectories for a given particle ID or cell.
        Args:
            input_ID (int, np.integer, Cell): ID of the particle or a Cell object.

        Returns:
            list: List of Cell objects in time order representing the trajectory of the particle.
        """
        if hasattr(self, 'tracked_centroids'):
            t = self.tracked_centroids

            if isinstance(input_ID, (int, np.integer)):  # If the input is already a particle ID, proceed
                particle_ID = input_ID
            elif isinstance(input_ID, Cell):  # If input is a Cell object, find its corresponding particle ID
                particle_ID = t[(t['frame'] == input_ID.frame) & (t['cell_number'] == input_ID.n)]['particle'].iloc[0]
            elif input_ID is None:  # If no input is provided, return an empty list
                return []
            else:
                raise ValueError(
                    f'Invalid input for get_particle function {input_ID}. Please provide a valid particle ID or Cell object.'
                )

            cell_rows = t[t.particle == particle_ID][['frame', 'cell_number']]  # Locate particle

            # Build list of particle appearances over time
            particle = []
            for frame_number, cell_number in np.array(cell_rows):
                particle.append(self.frames[frame_number].cells[cell_number])
        else:
            if isinstance(input_ID, (int, np.integer)):
                raise ValueError(f"No tracking data available, can't fetch particle {input_ID}.")
            elif isinstance(input_ID, Cell):
                particle = [input_ID]

        return particle

    def get_particle_attr(self, input_ID, attributes, fill_value=None):
        """
        Retrieve a specific attribute for a given particle ID or cell.

        Args:
            input_ID (int, np.integer, Cell): ID of the particle or a Cell object.
            attr (str): Attribute to retrieve.

        Returns:
            list: List of attribute values for the particle trajectory.
        """
        ret = []
        particle = self.get_particle(input_ID)
        for cell in particle:
            if isinstance(attributes, str):
                try:
                    ret.append(getattr(cell, attributes))
                except AttributeError:
                    if fill_value is not None:
                        ret.append(fill_value)
                    else:
                        raise AttributeError(f'Cell {cell.n} in frame {cell.frame} does not have attribute {attributes}')
            else:
                cell_attrs = []
                for attribute in attributes:
                    try:
                        cell_attrs.append(getattr(cell, attribute))
                    except AttributeError:
                        if fill_value is not None:
                            cell_attrs.append(fill_value)
                        else:
                            raise AttributeError(f'Cell {cell.n} in frame {cell.frame} does not have attribute {attribute}')
                ret.append(cell_attrs)
        return ret

    def set_particle_attr(self, input_ID, attributes, values):
        """
        Set a specific attribute for a given particle ID or cell.

        Args:
            input_ID (int, np.integer, Cell): ID of the particle or a Cell object.
            attr (str): Attribute to set.
            value (any): Value to set the attribute to.

        Returns:
            list: List of attribute values for the particle trajectory.
        """
        particle = self.get_particle(input_ID)
        if len(particle) != len(values):
            raise ValueError('Length of values must match the number of cells in the particle')
        for n, cell in enumerate(particle):
            if isinstance(attributes, str):
                setattr(cell, attributes, values[n])
            else:
                for attribute, value in zip(attributes, values[n]):
                    setattr(cell, attribute, value)

    def get_tracking_row(self, frame_or_cell: Union[int, Cell], cell_number: Optional[int] = None, particle_number: Optional[int] = None):
        """
        Retrieve the tracking data for a given cell at a given frame.

        Args:
            frame_or_cell (int | Cell): Frame number or a Cell object.
            cell_number (int): Cell number.
            particle_number (int, optional): Particle number.

        Returns:
            DataFrame: Tracking data for the cell at the given frame.
        """
        if not hasattr(self, 'tracked_centroids'):
            raise ValueError('No tracking data available.')
            
        if isinstance(frame_or_cell, Cell):
            cell=frame_or_cell
            frame_number = cell.frame
            cell_number = cell.n
        elif isinstance(frame_or_cell, int):
            frame_number = frame_or_cell
        
        if cell_number is None and particle_number is None:
            raise ValueError('Either cell_number or particle_number must be provided.')
        t = self.tracked_centroids
        if particle_number is not None:
            return t[(t['frame'] == frame_number) & (t['particle'] == particle_number)]
        else:
            return t[(t['frame'] == frame_number) & (t['cell_number'] == cell_number)]

    def get_velocities(self, from_csv=False, to_csv=False, **tracking_kwargs):
        """
        Calculate velocities from tracked centroids and optionally save/load from CSV.

        Args:
            from_csv (bool, optional): If True, loads velocities from a CSV file. Defaults to False.
            to_csv (bool, optional): If True, saves velocities to a CSV file. Defaults to False.
            **kwargs: Additional keyword arguments passed to track_centroids function if re-tracking is needed.

        Returns:
            DataFrame: DataFrame containing velocities and related data.
        """
        import pandas as pd

        # Load velocities from CSV if requested
        if from_csv:
            try:
                self.velocities = pd.read_csv(self.name.replace('segmented', 'tracking')[:-1] + '.csv', index_col=0)
                return self.velocities
            except FileNotFoundError:
                print(
                    "Couldn't find a file at {} to read tracking data from. Retracking.".format(
                        self.name.replace('segmented', 'tracking')[:-1] + '.csv'
                    )
                )

        # Check if tracking data is available and re-track if necessary
        if hasattr(self, 'tracked_centroids') and tracking_kwargs == {}:
            pass
        else:
            self.track_centroids(**tracking_kwargs)

        # Sort a copy of tracking data by particle and frame
        velocities_df = self.tracked_centroids.copy().sort_values(['particle', 'frame'])

        # Calculate displacements
        v = self.tracked_centroids[['x', 'y', 'frame', 'particle']].groupby('particle').apply(np.diff, axis=0, append=np.nan)
        velocities_df[['dx', 'dy', 'dt']] = np.concatenate(v.values)[:, :3]
        velocities_df[['dx', 'dy']] = velocities_df[['dx', 'dy']].div(velocities_df['dt'], axis=0)

        # Calculate velocity magnitude and direction
        velocities_df['v'] = np.linalg.norm(velocities_df[['dx', 'dy']], axis=1)
        velocities_df['theta'] = np.arctan2(velocities_df['dy'], velocities_df['dx'])

        # Store velocities
        self.velocities = velocities_df

        # Save velocities to CSV if requested
        if to_csv:
            print('Writing velocity data to {}.'.format(self.name.replace('segmented', 'tracking')[:-1] + '.csv'))
            csv_path = Path(self.name.split('/segmented/')[0] + '/tracking/')
            csv_path.makedir(exist_ok=True, parents=True)
            self.velocities.to_csv(self.name.replace('segmented', 'tracking')[:-1] + '.csv')
        return self.velocities

    # -------------Cell Cycle-------------
    def propagate_FUCCI_labels(self, progress=lambda x: x):
        """
        propagates FUCCI data forward in time by copying the last observed value.
        """
        t = self.tracked_centroids.copy().sort_values(['frame', 'cell_number'])

        # get all cell cycle data
        cell_cycles = []
        for frame in progress(self.frames):
            cell_cycles.extend(frame.get_cell_attrs('cycle_stage'))
        t['cell_cycle'] = cell_cycles

        # identify cells with cell cycle labels
        to_propagate = t.groupby('particle')['cell_cycle'].max() != 0
        to_propagate = to_propagate[to_propagate].index

        for particle in to_propagate:
            cell_cycle = t[t.particle == particle]['cell_cycle'].values
            propagated_cell_cycle = np.maximum.accumulate(cell_cycle)
            self.set_particle_attr(particle, 'cycle_stage', propagated_cell_cycle)

    def measure_FUCCI_by_transitions(self, green_threshold=-10, G2_peak_prominence=0.005, progress=lambda x: x):
        from scipy.signal import find_peaks

        def rolling_smooth(arr, window_size):
            if window_size % 2 == 0:
                raise ValueError('Window size must be odd')
            # Create window weights
            window = np.ones(window_size) / window_size

            # Use 'same' mode to preserve array size and edge padding
            smoothed = np.convolve(arr, window, mode='same')

            # Handle edge effects by using smaller windows at the edges
            half_window = window_size // 2
            for i in range(half_window):
                smoothed[i] = np.mean(arr[max(0, i - half_window) : i + half_window + 1])
                smoothed[-(i + 1)] = np.mean(arr[-(i + half_window + 1) :])

            return smoothed

        for frame in progress(self.frames):
            if not hasattr(frame.cells[0], 'red_intensity'):
                frame.get_red_green_intensities()

        for particle in progress(np.unique(self.tracked_centroids['particle'])):
            cells = self.get_particle(particle)
            if len(cells) < 15:
                continue
            red, green = np.array(self.get_particle_attr(particle, ['red_intensity', 'green_intensity'])).T

            green_gradient = np.gradient(green)
            G1_threshold = min(green_threshold, green_gradient.min() * 0.9)  # normalize somehow
            G1_transition = np.where(green_gradient[:-3] <= G1_threshold)[0]

            if len(G1_transition) > 0:
                G1_transition = G1_transition[0] + 1
                green_G1 = green[:G1_transition]
                green_SG2 = green[G1_transition:]
                if len(green_G1) > 5:
                    green_G1 = rolling_smooth(green_G1, 5)
                if len(green_SG2) > 5:
                    green_SG2 = rolling_smooth(green_SG2, 5)
                smoothed_green = np.concatenate([green_G1, green_SG2])
            else:
                G1_transition = -1
                green_G1 = green
                smoothed_green = rolling_smooth(green, 5)

            smoothed_red = rolling_smooth(red, 7)
            difference = smoothed_green / smoothed_red

            if G1_transition >= 0:
                S_phase = difference[G1_transition + 5 :]  # 5 frame offset to avoid G1 noise (should normalize to time scale)
                offset = G1_transition + 5
            else:
                S_phase = difference
                offset = 0
            candidates, properties = find_peaks(-S_phase, prominence=0)
            if len(candidates) > 0:
                biggest_peak = np.argmax(properties['prominences'])
                G2_prominence = properties['prominences'][biggest_peak]
                if G2_prominence < G2_peak_prominence:
                    G2_transition = -1
                else:
                    G2_transition = candidates[biggest_peak] + offset
            else:
                G2_transition = -1
                G2_prominence = 0

            if G1_transition >= 0 or G2_transition >= 0:
                for cell in cells:
                    cell.cycle_stage = 2
                if G1_transition >= 0:
                    for cell in cells[:G1_transition]:
                        cell.cycle_stage = 1
                if G2_transition >= 0:
                    for cell in cells[G2_transition:]:
                        cell.cycle_stage = 3

    def get_interpretable_FUCCI(self, zero_remainder=True, impute_zeros=True):
        from .FUCCI_linking import (
            correct_daughter_G1,
            correct_mother_G2,
            get_problematic_IDs,
            impute_fill,
            remove_sandwich_flicker,
            smooth_small_flickers,
        )

        cell_cycle = np.concatenate([frame.fetch_cell_cycle() for frame in self.frames])
        if hasattr(self, 'tracked_centroids') and len(self.tracked_centroids) == len(cell_cycle):
            t = self.tracked_centroids.sort_values(['frame', 'cell_number'])
        else:
            t = self.track_centroids().sort_values(['frame', 'cell_number'])
        t['cell_cycle'] = np.concatenate([frame.fetch_cell_cycle() for frame in self.frames])

        for ID in get_problematic_IDs(t):
            cell_cycle = t.loc[t.particle == ID, 'cell_cycle'].copy()

            if impute_zeros:
                # simple fix: extend existing data into any zero regions
                cell_cycle = impute_fill(cell_cycle, limit=6)

            # simple fix: smooth single-frame flickers by neighbor values
            cell_cycle = smooth_small_flickers(cell_cycle)

            # manual check: if a cell briefly goes red after G2 and then vanishes, it's just a byproduct of changes before mitosis
            cell_cycle = correct_mother_G2(cell_cycle)

            # manual check: similarly, if a cell starts out orange or red and then goes green, it's just vestigial fluorescence from mitosis.
            cell_cycle = correct_daughter_G1(cell_cycle)

            # manual check: if a cell goes from one phase to another and back, it's probably spurious (especially if out of temporal order).
            # the order of these checks matters (especially 3,2,3 then 2,3,2) because lots of cells will flicker and which check goes first determines who wins. Sorted by descending observed frequency
            cell_cycle = remove_sandwich_flicker(cell_cycle, [1, 3, 1])  # G1 jumps to G2
            cell_cycle = remove_sandwich_flicker(
                cell_cycle, [3, 2, 3]
            )  # G2 back to S. This happens all the time because green creeps into red
            cell_cycle = remove_sandwich_flicker(
                cell_cycle, [1, 3, 2]
            )  # G1 to S *through* G2. This is also likely because red replaces green
            cell_cycle = remove_sandwich_flicker(
                cell_cycle, [1, 0, 2]
            )  # G1 to S through NS. This is also likely because red replaces green

            cell_cycle = remove_sandwich_flicker(cell_cycle, [2, 3, 2])

            t.loc[t.particle == ID, 'cell_cycle'] = np.array(cell_cycle)

        if zero_remainder:
            for ID in get_problematic_IDs(t):
                t.loc[t.particle == ID, 'cell_cycle'] = 0  # remove any FUCCI data that still doesn't make sense

        for frame_number, cell_cycle_stages in t.groupby('frame')['cell_cycle']:
            self.frames[frame_number].write_cell_cycle(cell_cycle_stages)

        self.tracked_centroids['cell_cycle'] = t['cell_cycle']  # also export cell cycle info to tracking

        return t

    # -------------Cell Divisions-------------
    def get_mitoses(self, persistence_threshold=0, distance_threshold=1.5, weights=None, biases=None, score_cutoff=1, **kwargs):
        """
        Detect potential mitotic events in the cell tracking data.

        Args:
            persistence_threshold (int, optional): Minimum duration daughter cells must be present after division. Defaults to 0.
            distance_threshold (float, optional): Maximum distance threshold for considering a cell division. If not provided, it is automatically set based on cell area statistics.
            biases (array-like, optional): Biases used in mitosis scoring. Defaults set using manually labeled MDCK data.
            weights (array-like, optional): Weights used in mitosis scoring. Defaults set using manually labeled MDCK data.
            score_cutoff (int, optional): Threshold for considering a detected mitosis event. Higher is more forgiving. Defaults to 1.

        Returns:
            list: List of potential mitotic events, each represented as a DataFrame containing information about mother and daughter cells.
        """
        from .mitosis_detection import get_mitosis_scores, get_viable_mitoses, threshold_mitoses

        if not hasattr(self, 'tracked_centroids'):
            self.track_centroids(**kwargs)

        mitosis_df = get_viable_mitoses(self, distance_threshold=distance_threshold, persistence_threshold=persistence_threshold)

        self.mitosis_scores = get_mitosis_scores(mitosis_df, weights=weights, biases=biases)
        self.mitoses = threshold_mitoses(self.mitosis_scores, threshold=score_cutoff)
        self.mitoses = self.mitoses.astype(
            {'mother': 'Int64', 'daughter1': 'Int64', 'daughter2': 'Int64'}
        )  # nullable integer columns
        return self.mitoses

    def add_mitosis(self, m, compute_scores=False):
        """
        Add a mitosis event to the TimeStack object.

        Args:
            m (DataFrame): DataFrame containing information about the mitotic event.
            compute_scores (bool, optional): If True, computes mitosis scores for the added event. Defaults to False (score is NaN).
            TODO: add mitosis score computation
        """
        if not hasattr(self, 'mitoses'):
            self.mitoses = pd.DataFrame(columns=['frame', 'mother', 'daughter1', 'daughter2', 'score'])
            self.mitoses = self.mitoses.astype(
                {'frame': int, 'mother': 'Int64', 'daughter1': 'Int64', 'daughter2': 'Int64', 'score': float}
            )  # nullable integer columns
            self.mitosis_scores = self.mitoses.copy()

        new_idx = self.mitosis_scores.index.max() + 1 if len(self.mitosis_scores) > 0 else 0
        self.mitoses.loc[new_idx] = m
        self.mitosis_scores.loc[new_idx] = m
        return self.mitoses


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


class SegmentedImage:
    """
    A class for storing and manipulating segmented image data.
    One SegmentedImage object contains data for a single image frame, including masks, outlines, and cell objects.
    """

    verbose = False

    def vprint(self, obj):
        if self.verbose:
            print(obj)
        else:
            pass

    def __init__(self, data, name=None, frame_number=None, verbose=False, scale=None, units=None, **kwargs):
        """
        Initializes an Image object.
        The data can be passed either as a dictionary or as a path to a _seg.npy file containing the data.
        _seg.npy files are generated either by segmentation_tools or from the Cellpose GUI.

        Args:
            data (str or dict): Path to a _seg.npy file or a dictionary containing the segmented image data.
            name (str, optional): Name of the image, which will be used during exporting. Defaults to the loaded file path.
            frame_number (int, optional): Frame number for the image sequence if loaded in a SegmentedStack.
            verbose (bool, optional): If True, prints debug statements. Defaults to False.
            scale (float, optional): XY scale factor for converting pixels to desired units.
            units (str, optional): Units for measurements.
        """

        if isinstance(data, str):  # if a string is passed, assume it's a path to a seg.npy file containing a dictionary
            from segmentation_tools.io import load_seg_npy

            # pull load_seg_npy parameters from kwargs
            name = data
            load_seg_npy_kwargs = {key: kwargs.pop(key) for key in ['load_img', 'mend', 'max_gap_size'] if key in kwargs}
            data = load_seg_npy(data, **load_seg_npy_kwargs)
        elif isinstance(data, dict):  # already passed a dictionary
            pass
        else:
            raise ValueError(f'data must be a path to a seg.npy file or a dictionary of data from a seg.npy file, not {type(data)}.')

        if 'masks' not in data.keys():
            raise ValueError('Failed to instantiate SegmentedImage: segmented data must contain masks.')

        if 'outlines' not in data.keys():
            data['outlines'] = masks_to_outlines(data['masks'])

        # Print debug statements if verbose mode is enabled
        self.verbose = verbose

        for key, value in kwargs.items():  # Set any additional attributes passed as keyword arguments
            setattr(self, key, value)

        # Set pixel units and scale (if specified)
        if units is not None:
            self.units = units  # Units for measurements
        if scale is not None:
            self.scale = scale  # Scale factor for converting pixels to desired units
        self.frame_number = frame_number  # Frame number for the image sequence

        # Load the dictionary data into the Image object
        for attr in set(data.keys()):
            setattr(self, attr, data[attr])

        if name is None:
            try:
                name = data['name']
            except KeyError:
                raise ValueError('SegmentedImage data must contain a name attribute or a name must be provided.')

        self.name = name

        # set masks datatype
        max_mask = self.masks.max()
        self.masks = fastremap.refit(self.masks, value=max_mask * 2)  # refit to a larger datatype to avoid overflow errors

        self.resolution = self.masks.shape  # Image resolution

        try:
            self.outlines = self.outlines.todense()
        except AttributeError:  # if it's already dense, carry on
            pass
        self.outlines = self.outlines != 0  # convert to boolean

        # Generate outlines_list or load from file
        if not hasattr(self, 'outlines_list'):
            self.vprint('creating new outlines_list from masks')
            self.outlines_list = outlines_list(self.masks)  # Generate outlines_list

        cells = fastremap.unique(self.masks)
        cells = cells[cells != 0] - 1
        self.n_cells = len(cells)  # Number of detected cells in field of view (FOV)
        if self.n_cells != max_mask:
            print(f'WARNING: {self.name} masks are not contiguous. Renumbering...')
            fastremap.renumber(self.masks, in_place=True)

        # Instantiate Cell objects for each cell labeled in the image
        self.cells = np.array(
            [Cell(n, self.outlines_list[n], parent=self) for n in range(self.n_cells)]
        )

        # assign cell cycle to cell objects
        if hasattr(self, 'cell_cycles'):
            if len(self.cell_cycles) != len(cells):
                print(
                    f'WARNING: {self.name} cell cycle data does not match the number of cells in the image: {self.n_cells} cells in the image, {len(self.cell_cycles)} cell cycle values.'
                )
                self.cell_cycles = self.cell_cycles[cells]
            self.set_cell_attrs('cycle_stage', self.cell_cycles)

        if hasattr(self, 'heights'):
            if not hasattr(self, 'zero_to_nan'):
                self.zero_to_nan = True
            self.to_heightmap(zero_to_nan=self.zero_to_nan)

    # -------------Convert to HeightMap-------------
    def to_heightmap(self, **kwargs):
        """converts the image to a height map."""
        self.__class__ = HeightMap
        self.zero_to_nan = kwargs.pop('zero_to_nan', True)  # default to converting zeros to nans

    # -------------I/O-------------
    def load_img(self, normalize=False):
        self.img = np.load(self.name, allow_pickle=True).item()['img']
        if normalize:
            self.img = preprocessing.normalize(self.img, dtype=np.float32)  # normalize

    def to_seg_npy(self, export_path=None, overwrite_img=False, write_attrs=[]):
        try:
            data = np.load(self.name, allow_pickle=True).item()

            if not overwrite_img and 'img' in data.keys():  # take unmodified img from the original file
                img = data['img']
            else:  # if img is not in the data or we're overwriting it
                img = self.img
        except FileNotFoundError:  # if the file doesn't exist, we'll just use the img we have
            img = self.img
        except Exception as e:
            print(f'Error loading {self.name}: {e}. Using current img data.')
            img = self.img

        _outlines_list = [cell.outline for cell in self.cells]

        export = {'img': img, 'masks': self.masks, 'outlines': self.outlines, 'outlines_list': _outlines_list}

        optional_attrs = [
            'FUCCI',
            'cell_cycles',
            'volumes',
            'scale',
            'z_scale',
            'units',
            'heights',
            'coverslip_height',
        ]  # if any of these exist, export them as well
        write_attrs = set(write_attrs + optional_attrs)  # add optional attrs to write_attrs
        for attr in write_attrs:
            try:
                export[attr] = getattr(self, attr)
            except AttributeError:
                continue

        if export_path is None:  # if no export path is given, overwrite existing file
            export_path = self.name

        Path(export_path).parent.mkdir(parents=True, exist_ok=True)  # make sure the directory exists
        np.save(export_path, export)  # write segmentation file

    # -------------Mask Operations-------------
    def rot90(self, k=1):
        from .utils import rotate_points

        """ rotates the image 90 degrees k times."""
        self.masks = np.rot90(self.masks, k)
        self.outlines = np.rot90(self.outlines, k)
        for attr in ['img', 'heights']:
            if hasattr(self, attr):
                attr_array = getattr(self, attr)
                setattr(self, attr, np.rot90(attr_array, k))

        for cell in self.cells:
            if hasattr(cell, 'outline'):
                cell.outline = rotate_points(cell.outline, self.resolution[::-1], k)
                cell.get_centroid()

        if k % 2 == 1:
            self.resolution = self.resolution[::-1]

    def mend_gaps(self, gap_size=None):
        if gap_size is None:
            gap_size = self.mean_cell_area(scaled=False) / 2

        new_masks, mended = preprocessing.mend_gaps(self.masks, gap_size)

        if mended:
            changed_cells = np.unique(new_masks[new_masks != self.masks])
            changed_masks = np.zeros_like(self.masks, dtype=int)
            changed_masks_bool = np.isin(new_masks, changed_cells)
            changed_masks[changed_masks_bool] = new_masks[changed_masks_bool]
            if self.has_outlines:
                _outlines_list = outlines_list(changed_masks)
                for cell, o in zip(self.cells[changed_cells], _outlines_list):
                    cell.outline = o
                    cell.get_centroid()

            self.masks = new_masks
            self.outlines = masks_to_outlines(self.masks)
        return mended

    def renumber_cells(self):
        """
        Renumbers cell masks, sorted by (y,x).
        Also rearranges self.cells to match the new order.
        """
        self.masks, order = fastremap.renumber(self.masks)
        order.pop(0)  # remove the 0 key

        cell_order = np.zeros(len(self.cells), dtype=int)
        for entry, key in order.items():
            cell_order[key - 1] = entry
        cell_order -= 1

        self.cells = self.cells[cell_order]
        self.set_cell_attrs('n', range(self.n_cells))  # reassign cell.n values
        return cell_order

    def add_outline(self, mask):
        outline = outlines_list(mask)[0]
        self.outlines[outline[:, 1], outline[:, 0]] = True

        return outline

    def merge_cells(self, cell_n1, cell_n2):
        # edit frame.masks, frame.outlines
        mask_2 = self.masks == cell_n2 + 1

        self.masks[mask_2] = cell_n1 + 1  # merge masks

        merged_mask = self.masks == cell_n1 + 1
        self.outlines[merged_mask] = False  # remove both outlines
        outline = self.add_outline(merged_mask)  # add merged outline

        # edit merged cell object
        new_cell = self.cells[cell_n1]
        new_cell.outline = outline

        if hasattr(new_cell, '_centroid'):
            del new_cell._centroid

        # remove cell 2
        idx = self.delete_cells([cell_n2])

        return new_cell, idx

    def delete_cells(self, cell_numbers):
        """deletes a cell from the image by number."""
        cell_numbers = [n for n in cell_numbers if n in self.masks]
        if len(cell_numbers) < 5:  # small number of cells, iterative approach
            to_clear = np.zeros(self.masks.shape, dtype=bool)
            for cell_number in cell_numbers:
                to_clear |= self.masks == cell_number + 1
        else:  # large number, use np.isin
            to_clear = np.isin(self.masks, np.array(cell_numbers) + 1)

        # remove mask and outlines
        self.masks[to_clear] = 0
        self.outlines[to_clear] = False

        idx = np.setdiff1d(np.arange(self.n_cells), cell_numbers)
        self.n_cells -= len(cell_numbers)
        self.cells = self.cells[idx]
        self.set_cell_attrs('n', np.arange(self.n_cells))

        remapping = fastremap.component_map(idx + 1, np.arange(len(idx)) + 1)
        remapping[0] = 0
        self.masks = fastremap.remap(self.masks, remapping)

        return idx

    def remove_cell(self, cell: Cell, mask: Optional[np.ndarray] = None):
        """removes a cell from the image by object."""
        if mask is None:
            mask = self.masks == cell.n + 1

        self.masks[mask] = 0
        self.outlines[mask] = False
        self.cells = np.delete(self.cells, cell.n)

        if cell.n < self.n_cells:
            for later_cell in self.cells[cell.n :]:
                later_cell.n -= 1
            self.masks[self.masks > cell.n + 1] -= 1

        self.n_cells -= 1

    def find_edge_cells(self, margin=1):
        """finds masks that are within some number of pixels from the edge of the image."""
        top = self.masks[:margin, :].flatten()
        bottom = self.masks[-margin:, :].flatten()
        left = self.masks[margin:-margin, :margin].flatten()
        right = self.masks[margin:-margin, -margin:].flatten()

        edge_cells = np.unique(np.concatenate([top, bottom, left, right]))
        edge_cells = edge_cells[edge_cells != 0] - 1

        return edge_cells

    def remove_edge_cells(self, margin=1):
        """removes masks that are within some number of pixels from the edge of the image."""
        edge_cells = self.find_edge_cells(margin)
        self.delete_cells(edge_cells)
        return edge_cells

    def add_cell(self, cell: Cell, mask: np.ndarray):
        """adds a cell to the image."""
        if cell.n < self.n_cells:  # need to renumber existing cells
            for later_cell in self.cells[cell.n:]:
                later_cell.n += 1
            self.masks[self.masks > cell.n] += 1
        else:
            cell.n = self.n_cells

        self.cells = np.insert(self.cells, cell.n, cell)
        self.n_cells += 1
        self.masks[mask] = cell.n + 1
        self.outlines[cell.outline[:, 1], cell.outline[:, 0]] = True

    # ------------FUCCI----------------
    def get_red_green_intensities(self, percentile=90, blur_sigma=4):
        def nth_percentile(x):
            return np.percentile(x, percentile)

        red_channel = self.img[:, :, 0]
        green_channel = self.img[:, :, 1]

        # gaussian blur
        red_channel = ndimage.gaussian_filter(red_channel, sigma=blur_sigma)
        green_channel = ndimage.gaussian_filter(green_channel, sigma=blur_sigma)

        red_frame = ndimage.labeled_comprehension(
            red_channel,
            default=0,
            out_dtype=np.uint16,
            func=nth_percentile,
            labels=self.masks,
            index=np.arange(1, self.masks.max() + 1),
        )
        green_frame = ndimage.labeled_comprehension(
            green_channel,
            default=0,
            out_dtype=np.uint16,
            func=nth_percentile,
            labels=self.masks,
            index=np.arange(1, self.masks.max() + 1),
        )

        self.set_cell_attrs(['red_intensity', 'green_intensity'], [red_frame, green_frame])

    def measure_FUCCI(
        self,
        percent_threshold=0.15,
        red_fluor_threshold=None,
        green_fluor_threshold=None,
        orange_brightness=1.5,
        threshold_offset=0,
        noise_score=0.02,
    ):
        """
        Measure the FUCCI (Fluorescence Ubiquitination Cell Cycle Indicator) levels in cells.

        Args:
            use_masks (bool): If True, use masks for measurements.
            percent_threshold (float): Threshold for percentage comparison.
            red_fluor_threshold (float): Threshold for red fluorescence.
            green_fluor_threshold (float): Threshold for green fluorescence.

        Returns:
            tuple: Red fluorescence percentages, Green fluorescence percentages.
        """
        # Ensure FUCCI data is available
        if not hasattr(self, 'FUCCI'):  # Check if FUCCI data exists
            print('No FUCCI channel found. Using image data for FUCCI measurements.')
            self.FUCCI = np.array([self.img[..., 0], self.img[..., 1]])  # Use image data for FUCCI measurements
        nuclear_size_threshold = np.median(self.cell_areas(scaled=False)) * percent_threshold
        red, green = self.FUCCI

        # threshold fluorescence levels
        if not red_fluor_threshold:
            red_fluor_threshold = (
                preprocessing.get_fluor_threshold(red, nuclear_size_threshold, noise_score=noise_score) + threshold_offset
            )
        if not green_fluor_threshold:
            green_fluor_threshold = (
                preprocessing.get_fluor_threshold(green, nuclear_size_threshold, noise_score=noise_score) + threshold_offset
            )
        self.red_fluor_threshold = red_fluor_threshold
        self.green_fluor_threshold = green_fluor_threshold
        self.orange_brightness = orange_brightness
        thresholded_green = green > green_fluor_threshold
        thresholded_red = red > red_fluor_threshold
        thresholded_orange = (red > red_fluor_threshold * orange_brightness) & thresholded_green

        fluor_percentages = np.stack(
            [
                preprocessing.fluorescent_percentages(self.masks, thresholded_green),
                preprocessing.fluorescent_percentages(self.masks, thresholded_red),
                preprocessing.fluorescent_percentages(self.masks, thresholded_orange),
            ],
            axis=1,
        )

        fluor_nuclei = fluor_percentages > percent_threshold
        fluor_nuclei[fluor_nuclei[:, 2], :2] = 0  # wherever cells are orange, turn off red and green

        red_or_green = fluor_nuclei[:, 0] & fluor_nuclei[:, 1]
        fluor_nuclei[red_or_green, np.argmin(fluor_percentages[red_or_green, :2], axis=1)] = (
            0  # if not orange, pick red or green based on which has a higher percentage
        )
        cc_stage_number = np.argmax(fluor_nuclei, axis=1) + 1  # G1->1, S->2, G2->3
        cc_stage_number[np.sum(fluor_nuclei, axis=1) == 0] = 0  # no signal->0
        self.cell_cycles = cc_stage_number
        for cell, cycle_stage in zip(self.cells, cc_stage_number):
            cell.cycle_stage = cycle_stage
        return cc_stage_number

    # -------------Metrics---------------
    # generalized methods for getting and setting cell attributes
    def get_cell_attrs(self, attributes, fill_value=None):
        ret = []
        for cell in self.cells:
            if isinstance(attributes, str):
                try:
                    ret.append(getattr(cell, attributes))
                except AttributeError:
                    if fill_value is not None:
                        ret.append(fill_value)
                    else:
                        raise AttributeError(f'Cell {cell.n} does not have attribute {attributes}')
            else:
                cell_attrs = []
                for attribute in attributes:
                    try:
                        cell_attrs.append(getattr(cell, attribute))
                    except AttributeError:
                        if fill_value is not None:
                            cell_attrs.append(fill_value)
                        else:
                            raise AttributeError(f'Cell {cell.n} does not have attribute {attribute}')
                ret.append(cell_attrs)
        return ret

    def set_cell_attrs(self, attributes, values):
        # TODO: make sure values and cells are same length
        if isinstance(attributes, str):  # if only one attribute is being set
            self.set_cell_attr(attributes, values)
        else:  # if multiple attributes are being set
            for attribute, val in zip(attributes, values):
                self.set_cell_attr(attribute, val)

    def set_cell_attr(self, attribute, values):
        if len(values) != len(self.cells):
            raise ValueError(
                f'Error setting attribute {attribute}: Length of values ({len(values)}) must match the number of cells ({len(self.cells)}) in the image'
            )
        for cell, value in zip(self.cells, values):
            setattr(cell, attribute, value)

    def cell_areas(self, scaled=True):
        """
        Calculate areas of individual cells in the image (area given as number of pixels).
        Args:
            scaled (bool, optional): If True, returns scaled areas considering image scaling. Defaults to True.
        Returns:
            numpy.ndarray: Array of cell areas.
        """
        # Find unique cell labels and their counts (n_pixels)
        areas = fastremap.unique(self.masks, return_index=False, return_counts=True)[1][1:]

        if scaled:  # If scaled option is True, scale the areas based on image scaling factor
            return areas * self.scale**2
        else:
            return areas  # Otherwise, return the unscaled areas

    def mean_cell_area(self, scaled=True):
        """get cell areas by counting number of pixels per mask"""
        mean_area = np.mean(self.cell_areas(scaled))
        return mean_area

    def FOV_area(self, scaled=True):
        """width*height drawn from resolution"""
        FOV_pixels = np.multiply(*self.resolution)
        if scaled:
            return FOV_pixels * self.scale**2
        else:
            return FOV_pixels

    def mean_cell_length(self, scaled=True):
        """characteristic length scale for particle tracking and mitosis detection"""
        return np.sqrt(self.mean_cell_area(scaled))

    @property
    def mean_density(self):
        """cells per square mm, measured by taking inverse of mean cell area"""
        return 1e6 / self.mean_cell_area()  # (1e6 mm^2/um^2)/(um^2 per cell)=cells per mm^2

    # -------------Centroids-------------
    def centroids(self):
        """returns an array of all cell centroids."""
        if len(self.cells) == 0:
            return np.empty((0, 2))  # no cells in this frame

        return np.array(self.get_cell_attrs('centroid'))

    # -------------Fit Ellipses to Cells-------------
    def fit_ellipses(self):
        """fits each cell outline to an ellipse, and assigns the angle and aspect ratio data to the corresponding Cell."""
        for cell in (
            self.cells
        ):  # can I vectorize fitting all these ellipses? I just don't really know how this works well enough to optimize I guess
            cell.fit_ellipse()

    def cell_ellipse_parameters(self):
        """fetches ellipse fitting data from each Cell and returns a DataFrame for all Cells in the Image."""
        try:  # see if all the theta and aspect ratio data is there to be spit out
            return pd.DataFrame([[cell.theta, cell.aspect] for cell in self.cells], columns=['theta', 'aspect'])

        except AttributeError:  # looks like at least some of it's missing, so generate it
            self.fit_ellipses()
            df = []
            for cell in self.cells:
                if hasattr(cell, 'theta') and hasattr(cell, 'aspect'):
                    df.append([cell.theta, cell.aspect])
            return pd.DataFrame(df, columns=['theta', 'aspect'])

    # -------------Vertex Reconstruction-------------
    def get_TCJs(self):
        """set Cell.vertices attribute for all cells with complete set of neighbors."""
        from scipy.signal import convolve2d

        # find good cells
        self.bad_cell_indices = (
            np.unique(self.masks[np.where(convolve2d(self.masks != 0, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])[1:-1, 1:-1] < 5)])[1:]
            - 1
        )  # indices of cells without full set of neighbors
        self.good_cell_indices = np.delete(
            np.arange(self.n_cells), self.bad_cell_indices
        )  # indices of cells with full set of neighbors (can reconstruct vertices from TCJs)

        if len(self.cells) == 0:
            self.TCJs = np.empty((0, 2))
            return self.TCJs

        # get bottom-right corners of every non-zero square of pixels
        cc_junctions = np.column_stack(np.where(convolve2d(self.outlines, np.array([[1, 1], [1, 1]])) == 4))

        # find 2x2 boxes with three or four cells in them (our metric for TCJs)
        def is_TCJ(x, y):
            return (
                len(np.unique(self.masks[x - 1 : x + 1, y - 1 : y + 1])) >= 3
            )  # very rare cases will have four-cell junctions, generally three

        is_TCJ_vect = np.vectorize(is_TCJ)
        TCJs = cc_junctions[is_TCJ_vect(cc_junctions[:, 0], cc_junctions[:, 1])]
        self.TCJs = TCJs

        # figure out which cells participate in each TCJ
        TCJ_cells = []
        for x, y in TCJs:
            TCJ_cells.append(
                self.masks[x - 1 : x + 1, y - 1 : y + 1].flatten()
            )  # kind of silly to do this again but idk how to vectorize and return a more complex output than True/False
        TCJ_cells = np.array(TCJ_cells) - 1  # subtract 1 so cells are 0-indexed
        self.TCJ_cells = TCJ_cells

        # sort TCJs into their constituent Cell objects
        incomplete_polygons = []
        for good_cell in [self.cells[n] for n in self.good_cell_indices]:
            good_cell.vertices = np.unique(
                TCJs[np.where(TCJ_cells == good_cell.n)[0]], axis=0
            )  # find full set of vertices for the nth cell
            good_cell.sorted_vertices = good_cell.sort_vertices()  # get the order in which vertices are connected
            if (
                len(good_cell.vertices) < 3
            ):  # find cells with 2 or fewer found vertices (line or point, not a polygon. Probably something went wrong)
                incomplete_polygons.append(good_cell.n)
                self.vprint('found an incomplete polygon at index {} of {}'.format(good_cell.n, self.name))
        if len(incomplete_polygons) > 0:
            self.good_cell_indices = self.good_cell_indices[
                ~np.any([self.good_cell_indices == x for x in incomplete_polygons], axis=0)
            ]  # drop incomplete polygons
            self.bad_cell_indices = np.append(
                self.bad_cell_indices, incomplete_polygons
            )  # for consistency, add incomplete polygons to our index of bad cells. Not using this rn.
        return self.TCJs

    def good_cells(self):
        """returns a list of the Cell objects with a full set of neighbors."""
        if not hasattr(self, 'good_cell_indices'):
            self.get_TCJs()
        return self.cells[self.good_cell_indices]

    def shape_parameters(self):
        """returns an array of all shape parameters."""
        shape_parameters = []
        for cell in self.good_cells():
            shape_parameters.append(cell.shape_parameter)
        return np.array(shape_parameters)

    def get_spherical_volumes(self):
        """returns the volumes of all cells in the image, assuming they are spherical."""
        areas = np.array(self.get_cell_attrs('area'))
        volumes = 4 / 3 * np.pi * (areas / np.pi) ** (3 / 2)
        self.set_cell_attrs('volume', volumes)
        return volumes

    def cell_polygons(self, ec='k', fc='lightsteelblue', linewidths=0.8, **kwargs):
        """returns a matplotlib PatchCollection of vertex-reconstructed cells for efficient plotting."""
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon

        cell_polygons = []
        for cell in self.good_cells():
            vertices = cell.sorted_vertices
            vertices = np.array([vertices[:, 1], vertices[:, 0]]).T  # flip x and y for plotting
            cell_polygons.append(Polygon(vertices))
        cell_collection = PatchCollection(cell_polygons, edgecolor=ec, facecolor=fc, linewidths=linewidths, **kwargs)
        return cell_collection


class HeightMap(SegmentedImage):
    nori_R = 8192
    lipid_density = 1.010101  # g/mL
    protein_density = 1.364256  # g/mL

    def __init__(self, seg_path, mesh_path=None, zero_to_nan=True, NORI=False, **kwargs):
        super().__init__(seg_path, **kwargs)
        if not hasattr(self, 'heights'):
            # for NoRI, or backward compatibility to MGX. New segmentations should have heights in them.
            if mesh_path is None:
                mesh_path = seg_path.replace('segmented', 'heights').replace('seg.npy', 'binarized.tif')
            self.heights, self.height_img = preprocessing.read_height_tif(mesh_path, zero_to_nan=zero_to_nan)

            if NORI:
                self.masks_3d = np.repeat(self.masks[np.newaxis], self.height_img.shape[0], axis=0)
                self.masks_3d[~self.height_img] = 0
                self.read_NORI()

        self.zero_to_nan = zero_to_nan

    @property
    def scaled_heights(self):
        """heights in um"""
        scaled = self.heights.copy().astype(float)

        if hasattr(
            self, 'coverslip_height'
        ):  # new convention is to store heights as ints, and subtract the coverslip height after
            scaled -= self.coverslip_height

        if self.zero_to_nan:
            scaled[scaled <= 0] = np.nan
        return scaled * self.z_scale

    def read_NORI(self, file_path=None, mask_nan_z=True):
        from skimage import io

        if file_path is None:
            file_path = self.name.replace('segmented', 'NORI').replace('seg.npy', 'NORI.tif')
        self.NORI = io.imread(file_path).astype(float) / self.nori_R
        self.NORI[..., 0] = self.NORI[..., 0] * self.protein_density
        self.NORI[..., 1] = self.NORI[..., 1] * self.lipid_density
        if mask_nan_z:
            # np.ma.masked_where(np.repeat(np.isnan(self.z)[np.newaxis], self.NORI.shape[0], axis=0), self.NORI, copy=False)
            self.NORI[:, np.isnan(self.heights)] = np.nan
        return self.NORI

    def get_NORI_density(self):
        """NORI density in g/mL"""
        if not hasattr(self, 'NORI'):
            self.read_NORI()
        self.NORI_density = np.array(
            [ndimage.mean(self.NORI[..., i], labels=self.masks_3d, index=range(1, self.masks.max() + 1)) for i in range(3)]
        )
        self.NORI_density_std = np.array(
            [
                ndimage.standard_deviation(self.NORI[..., i], labels=self.masks_3d, index=range(1, self.masks.max() + 1))
                for i in range(3)
            ]
        )
        self.set_cell_attrs('NORI_density', self.NORI_density.T)
        return self.NORI_density

    def get_NORI_mass(self):
        """NORI mass in picograms"""
        if not hasattr(self, 'NORI'):
            self.read_NORI()
        self.NORI_mass = np.array(
            [ndimage.sum(self.NORI[..., i], labels=self.masks_3d, index=range(1, self.masks.max() + 1)) for i in range(3)]
        )
        self.NORI_mass *= self.scale**2 * self.z_scale
        self.set_cell_attrs('NORI_mass', self.NORI_mass.T)
        return self.NORI_mass

    def get_volumes(self, scale=None, z_scale=None):
        """returns the volumes of all cells in the image."""
        if scale is not None:
            self.scale = scale
        if z_scale is not None:
            self.z_scale = z_scale

        self.volumes = ndimage.sum(self.scaled_heights, labels=self.masks, index=range(1, self.masks.max() + 1)) * self.scale**2
        self.mean_heights = ndimage.mean(self.scaled_heights, labels=self.masks, index=range(1, self.masks.max() + 1))
        self.set_cell_attrs('volume', self.volumes)
        self.set_cell_attrs('height', self.mean_heights)

        return self.volumes


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
