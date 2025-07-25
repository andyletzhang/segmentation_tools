import xml.etree.ElementTree as ET

import numpy as np
from nd2 import ND2File
from tifffile import TiffFile

from segmentation_tools.segmented_comprehension import SegmentedImage


class ND2:
    def __init__(self, path):
        self.path = path
        self.nd2 = ND2File(path)
        self.array = read_nd2(self.nd2)
        self.shape = read_nd2_shape(self.nd2, order='TPZYXC')
    
    @property
    def ndim(self):
        return len(self.shape)

    @staticmethod
    def fetch_placeholders(array):
        if isinstance(array, np.ndarray):
            dim_shape = array.shape
            out = np.array([img() for img in array.flatten()])
            if out[0].ndim == 3: # color image
                out=out.transpose(0, 2, 3, 1) # Move the channel axis to the end
            out = out.reshape(*dim_shape, *out[0].shape) # Reshape T, P, Z dimensions
        elif callable(array):
            out = array()
            if out.ndim == 3: # color image
                out = out.transpose(1, 2, 0) # Move the channel axis to the end
        else:
            raise ValueError(f'Invalid input type: {type(array)}')
        return out

    def substack(self, t=None, p=None, z=None):
        return self.array[t, p, z]

    def __getitem__(self, key):
        out = self.fetch_placeholders(self.array[key])
        return out

    def __iter__(self):
        if self.shape[:3] == (1, 1, 1):
            yield self[0, 0, 0]  # single image
        else:
            for img in np.squeeze(self.array):
                yield self.fetch_placeholders(img)

    def __len__(self):
        if self.shape[:3] == (1, 1, 1):
            return 1
        else:
            return np.squeeze(self.array).shape[0]

    def close(self):
        self.nd2.close()

    def open(self):
        self.nd2.open()

    def __del__(self):
        self.close()


# Function to convert XML elements into a dictionary
def xml_to_dict(element: ET.Element) -> dict:
    data_dict = {element.tag.split('}')[-1]: {} if element.attrib or list(element) else element.text}

    # Process children recursively
    for child in element:
        child_dict = xml_to_dict(child)
        data_dict[element.tag.split('}')[-1]].update(child_dict)

    # Process attributes
    if element.attrib:
        data_dict[element.tag.split('}')[-1]].update(('@' + k, v) for k, v in element.attrib.items())

    return data_dict


def read_ome_metadata(tif_file: TiffFile) -> dict:
    root = ET.fromstring(tif_file.ome_metadata)
    return xml_to_dict(root)['OME']


def read_tif_shape(tif_file: TiffFile) -> tuple:
    if tif_file.is_imagej:
        shape = []
        for key in ['frames', 'slices', 'channels']:
            if key in tif_file.imagej_metadata:
                shape.append(tif_file.imagej_metadata[key])
            else:
                shape.append(1)
        img_shape = list(tif_file.pages[0].shape)[:2]
        shape = shape + img_shape

        dimension_order = 'XYCZT'  # default ImageJ dimension order

    elif tif_file.is_ome:
        shape = []
        metadata_dict = read_ome_metadata(tif_file)
        for key in ['@SizeT', '@SizeZ', '@SizeC', '@SizeY', '@SizeX']:
            shape.append(int(metadata_dict['Image']['Pixels'][key]))

        dimension_order = metadata_dict['Image']['Pixels']['@DimensionOrder']

    else:  # no recognized metadata
        shape = (len(tif_file.pages), 1, 1) + tif_file.pages[0].shape  # assume simple time series
        dimension_order = 'XYCZT'

    return (1,) + tuple(shape), dimension_order + 'P'


def read_tif(tif_file: TiffFile) -> np.ndarray:
    shape, order = read_tif_shape(tif_file)
    tif_shape = tif_file.pages[0].shape
    if len(tif_shape) == 2:
        pages_dims = 'TPZC'
    elif len(tif_shape) == 3:
        pages_dims = 'TPZ'
    else:
        raise ValueError(f'Found invalid tiff shape when reading tif file: {tif_shape}')

    # shape without the XYC dimensions
    pages_shape = shape[: -len(tif_shape)]
    axes_map = {axis: i for i, axis in enumerate(reversed(order))}

    if len(tif_file.pages) == 1 and np.prod(pages_shape) > 1:
        # BigTIFF-adjacent (?) case with only one page.
        import dask.array as da

        tif_pages = da.from_zarr(tif_file.series[0].aszarr())
        tif_pages = tif_pages.reshape(shape).transpose(
            axes_map['T'], axes_map['P'], axes_map['Z'], axes_map['C'], axes_map['Y'], axes_map['X']
        )
        placeholder = np.empty(tif_pages.shape[:3], dtype=object)
        for i, j, k in np.ndindex(placeholder.shape):
            placeholder[i, j, k] = lambda i=i, j=j, k=k: np.array([a.compute() for a in tif_pages[i, j, k]])
        return placeholder
    else:
        tif_pages = np.array(tif_file.pages).reshape(pages_shape).transpose(*[axes_map[dim] for dim in pages_dims])
        placeholder = np.empty(tif_pages.shape[:3], dtype=object)
        for i, j, k in np.ndindex(placeholder.shape):
            if len(tif_shape) == 2:
                placeholder[i, j, k] = lambda i=i, j=j, k=k: np.array([img.asarray() for img in tif_pages[i, j, k]])
            else:
                placeholder[i, j, k] = lambda i=i, j=j, k=k: tif_pages[i, j, k].asarray()
        return placeholder


def read_nd2_shape(nd2_file: ND2File, order: str = 'TPZCYX') -> tuple:
    # Read metadata to get the shape (T, Z, C)
    default_order='TPZCYX'
    shape = [1, 1, 1, 1]
    for n, axis in enumerate(['T', 'P', 'Z', 'C']):
        if axis in nd2_file.sizes:
            shape[n] = nd2_file.sizes[axis]

    img_shape = nd2_file.shape[-2:]  # (Y, X)
    shape.extend(img_shape)  # Append (Y, X) to the shape

    # Reorder the shape according to the order
    shape = [shape[default_order.index(axis)] for axis in order]

    return tuple(shape)


def read_nd2(nd2_file: ND2File) -> np.ndarray:
    # Open ND2 file with ND2Reader
    shape = read_nd2_shape(nd2_file)
    placeholder_shape = (shape[0], shape[1], shape[2])  # Only (P, T, Z)
    n_images = shape[0] * shape[1] * shape[2]  # Total number of images

    # Lazy load structure (T, Z, C)
    placeholder = np.array([lambda i=i: nd2_file.read_frame(i) for i in range(n_images)]).reshape(placeholder_shape)

    return placeholder


# ----------Creating Segmentation Objects--------------------


def load_seg_npy(file_path, load_img=False, mend=False, max_gap_size=300):
    data = np.load(file_path, allow_pickle=True).item()

    if 'img' not in data.keys() and 'filename' in data.keys():
        # this seg.npy was made with the cellpose GUI
        data = convert_GUI_seg(data)

    if not load_img:
        del data['img']

    if mend:
        from segmentation_tools.preprocessing import mend_gaps

        data['masks'], mended = mend_gaps(data['masks'], max_gap_size)
        if mended:
            if 'outlines' in data.keys():
                del data['outlines']
            if 'outlines_list' in data.keys():
                del data['outlines_list']

    return data


def segmentation_from_img(img, name, **kwargs):
    shape = img.shape[:2]
    outlines = np.zeros(shape, dtype=bool)
    masks = np.zeros(shape, dtype=np.uint16)
    data = {'name': name, 'img': img, 'masks': masks, 'outlines': outlines}
    seg = SegmentedImage(data, **kwargs)
    return seg


def segmentation_from_zstack(zstack, name, **kwargs):
    shape = zstack.shape[1:3]
    outlines = np.zeros(shape, dtype=bool)
    masks = np.zeros(shape, dtype=np.uint16)
    data = {'name': name, 'zstack': zstack, 'img': zstack[0], 'masks': masks, 'outlines': outlines}
    seg = SegmentedImage(data, **kwargs)
    return seg


def convert_GUI_seg(seg, multiprocess=False, remove_edge_masks=True, mend=False, max_gap_size=20, export=False, out_path=None):
    """convert a segmentation image from the GUI to a format that can be used by the tracking algorithm"""
    from cellpose.utils import masks_to_outlines
    from skimage import io

    from segmentation_tools.preprocessing import mend_gaps, remove_edge_masks_tile

    img_path = seg['filename']
    try:
        img = io.imread(img_path)
    except FileNotFoundError:
        raise
    masks = seg['masks']
    if remove_edge_masks:
        if img.ndim == 2:
            membrane = img
        elif img.ndim == 3:
            color_channel = np.argmin(img.shape)
            membrane = img.take(-1, axis=color_channel)
        masks = remove_edge_masks_tile(membrane, masks)

    if mend:
        masks, _ = mend_gaps(masks, max_gap_size)

    if multiprocess:
        from cellpose.utils import outlines_list_multi

        outlines_list = outlines_list_multi(masks)
    else:
        from cellpose.utils import outlines_list

        outlines_list = outlines_list(masks)

    outlines = masks_to_outlines(masks)

    out_dict = {'img': img, 'masks': masks, 'outlines': outlines, 'outlines_list': outlines_list}
    if export:
        if out_path is None:
            out_path = seg.replace('.tif', '_seg.npy')

        if not out_path.endswith('seg.npy'):
            out_path += '_seg.npy'
        np.save(out_path, out_dict)

    return out_dict
