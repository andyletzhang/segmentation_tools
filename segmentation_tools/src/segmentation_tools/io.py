import xml.etree.ElementTree as ET
import numpy as np
from tifffile import TiffFile
from nd2 import ND2File

# Function to convert XML elements into a dictionary
def xml_to_dict(element):
    data_dict = {element.tag.split('}')[-1]: {} if element.attrib or list(element) else element.text}
    
    # Process children recursively
    for child in element:
        child_dict = xml_to_dict(child)
        data_dict[element.tag.split('}')[-1]].update(child_dict)
    
    # Process attributes
    if element.attrib:
        data_dict[element.tag.split('}')[-1]].update(('@' + k, v) for k, v in element.attrib.items())
    
    return data_dict

def read_ome_metadata(tif_file):
    root = ET.fromstring(tif_file.ome_metadata)
    return xml_to_dict(root)['OME']

def read_tif_shape(tif_file):
    if tif_file.is_imagej:
        shape=[]
        for key in ['frames','slices','channels']:
            if key in tif_file.imagej_metadata:
                shape.append(tif_file.imagej_metadata[key])
            else:
                shape.append(1)
        img_shape=list(tif_file.pages[0].shape)
        shape=shape+img_shape

        dimension_order='XYCZT' # default ImageJ dimension order

    elif tif_file.is_ome:
        shape=[]
        metadata_dict=read_ome_metadata(tif_file)
        for key in ['@SizeT','@SizeZ','@SizeC', '@SizeY', '@SizeX']:
            shape.append(int(metadata_dict["Image"]['Pixels'][key]))

        dimension_order=metadata_dict["Image"]['Pixels']['@DimensionOrder']

    else: # no recognized metadata
        shape=(len(tif_file.pages),1,1,)+tif_file.pages[0].shape # assume simple time series
        dimension_order='XYCZT'

    return tuple(shape), dimension_order

def read_tif(tif_file):
    shape, order=read_tif_shape(tif_file)
    axes_map = {axis: i for i, axis in enumerate(reversed(order))}
    reshaped=tuple(shape[axes_map[axis]] for axis in 'TZC')
    tif_pages=np.array(tif_file.pages).reshape(reshaped).transpose(axes_map['T'], axes_map['Z'], axes_map['C'])
    placeholder=np.empty((reshaped[0], reshaped[1]), dtype=object)
    for i in range(reshaped[0]):
        for j in range(reshaped[1]):
            placeholder[i,j]=lambda i=i, j=j:np.array([a.asarray() for a in tif_pages[i,j]]) # lazy load
    return placeholder # images in T, Z, C order

def read_nd2_shape(nd2_file):
    # Read metadata to get the shape (T, Z, C)
    shape = []
    is_v='P' in nd2_file.sizes and nd2_file.sizes['P'] > 1
    is_t='T' in nd2_file.sizes and nd2_file.sizes['T'] > 1

    multipoint_axis = ['P']
    if is_v and is_t:
        print('Warning: multipoint time-lapse data detected. Only the first time point will be loaded.')
    elif is_t:
        multipoint_axis=['T']

    for axis in multipoint_axis+['Z', 'C']:
        if axis in nd2_file.sizes:
            shape.append(nd2_file.sizes[axis])
        else:
            shape.append(1)  # Default to 1 if the axis is not in the ND2 file

    img_shape = nd2_file.shape[-2:]  # (Y, X)
    shape.extend(img_shape)  # Append (Y, X) to the shape
    return tuple(shape)

def read_nd2(nd2_file):
    # Open ND2 file with ND2Reader
    shape = read_nd2_shape(nd2_file)
    placeholder_shape = (shape[0], shape[1])  # Only (T, Z, C)
    n_images=shape[0]*shape[1]

    # Lazy load structure (T, Z, C)
    placeholder = np.array([lambda i=i: nd2_file.read_frame(i) for i in range(n_images)]).reshape(placeholder_shape)

    return placeholder

#----------Creating Segmentation Objects--------------------
from segmentation_tools.segmented_comprehension import SegmentedStack, SuspendedStack, TimeStack, SegmentedImage, HeightMap
    
def load_seg_npy(file_path, load_img=False, mend=False, max_gap_size=300):
    data=np.load(file_path, allow_pickle=True).item()
    
    if not 'img' in data.keys() and 'filename' in data.keys():
        # this seg.npy was made with the cellpose GUI
        data=convert_GUI_seg(data)

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
    shape=img.shape[:2]
    outlines=np.zeros(shape, dtype=bool)
    masks=np.zeros(shape, dtype=np.uint16)
    data={'name':name,'img':img,'masks':masks,'outlines':outlines}
    seg=SegmentedImage(data, **kwargs)
    return seg

def segmentation_from_zstack(zstack, name, **kwargs):
    shape=zstack.shape[1:3]
    outlines=np.zeros(shape, dtype=bool)
    masks=np.zeros(shape, dtype=np.uint16)
    data={'name':name,'zstack':zstack,'img':zstack[0],'masks':masks,'outlines':outlines}
    seg=SegmentedImage(data, **kwargs)
    return seg

def convert_GUI_seg(seg, multiprocess=False, remove_edge_masks=True, mend=False, max_gap_size=20, export=False, out_path=None):
    ''' convert a segmentation image from the GUI to a format that can be used by the tracking algorithm '''
    from cellpose.utils import masks_to_outlines
    from segmentation_tools.preprocessing import mend_gaps, remove_edge_masks_tile
    from skimage import io

    img_path=seg['filename']
    try:
        img=io.imread(img_path)
    except FileNotFoundError:
        raise
    masks=seg['masks']
    if remove_edge_masks:
        if img.ndim==2:
            membrane=img
        elif img.ndim==3:
            color_channel=np.argmin(img.shape)
            membrane=img.take(-1, axis=color_channel)
        masks=remove_edge_masks_tile(membrane, masks)

    if mend:
        masks, _ =mend_gaps(masks, max_gap_size)

    if multiprocess:
        from cellpose.utils import outlines_list_multi
        outlines_list=outlines_list_multi(masks)
    else:
        from cellpose.utils import outlines_list
        outlines_list=outlines_list(masks)

    outlines=masks_to_outlines(masks)

    out_dict={'img':img, 'masks':masks, 'outlines':outlines, 'outlines_list':outlines_list}
    if export:
        if out_path is None:
            out_path=seg.replace('.tif', '_seg.npy')
        
        if not out_path.endswith('seg.npy'):
            out_path+='_seg.npy'
        np.save(out_path, out_dict)
    
    return out_dict