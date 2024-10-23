import xml.etree.ElementTree as ET
import numpy as np
from tifffile import TiffFile

def get_nd2_zstack(nd2_file, v=0, **kwargs):
    ''' fetch a zstack from an ND2 file at the specified stage position.'''
    zstack=[]
    for i in range(nd2_file.sizes['z']):
        zstack.append(nd2_file.get_frame_2D(v=v, z=i, **kwargs))
    return np.array(zstack)

def get_nd2_RGB(nd2_file, v=0, z=0, **kwargs):
    RGB=[nd2_file.get_frame_2D(v=v, z=z, c=i, **kwargs) for i in range(3)]
    return np.array(RGB).transpose(1,2,0)

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

    else:
        raise ValueError('Unknown TIFF format (not saved as OME TIFF or ImageJ TIFF)')
    
    return tuple(shape), dimension_order

def read_tif(tif_path):
    tif_file=TiffFile(tif_path)
    shape, order=read_tif_shape(tif_file)
    axes_map = {axis: 4-i for i, axis in enumerate(order)}
    reshaped=[shape[axes_map[axis]] for axis in 'TZC']
    tif_pages=np.array(tif_file.pages).reshape(reshaped).transpose(axes_map['T'], axes_map['Z'], axes_map['C'])
    return tif_pages # images in T, Z, C order

def tiffpage_frame(tif_pages, v=0, z=0):
    img=tif_pages[v,z]
    if img.shape[0]==1:
        img=img[0].asarray()
    elif img.shape[0]>1:
        img=np.array([a.asarray() for a in img])
        if img.shape[0]!=3:
            print(f'Warning: unexpected number of color channels {img.shape[0]}')
        img=img.transpose(1,2,0)
    return img

def tiffpage_zstack(tif_pages, v=0):
    zstack=tif_pages[v]
    if zstack.shape[1]==1:
        return np.array([a.asarray() for a in zstack[:,0]]) # mono z stack
    else:
        shape=zstack.shape
        zstack=np.array([a.asarray() for a in zstack.flatten()]).reshape(shape).transpose(0,2,3,1)
        if zstack.shape[-1]!=3:
            print(f'Warning: unexpected number of color channels {zstack.shape[-1]}')
        return zstack

#----------Creating Segmentation Objects--------------------
from segmentation_tools.segmented_comprehension import SegmentedStack, SuspendedStack, TimeStack, SegmentedImage, HeightMap
def load_stack(stack_type='timelapse', **kwargs):
    if stack_type=='timelapse':
        return TimeStack(**kwargs)
    
    elif stack_type=='multipoint':
        return SegmentedStack(**kwargs)
    
    elif stack_type=='suspended':
        return SuspendedStack(**kwargs)

def stack_from_frames(frames, stack_type='timelapse', **kwargs):
    if stack_type=='timelapse':
        return TimeStack(from_frames=frames, **kwargs)
    
    elif stack_type=='multipoint':
        return SegmentedStack(from_frames=frames, **kwargs)
    
    elif stack_type=='suspended':
        return SuspendedStack(from_frames=frames, **kwargs)
    
def load_segmentation(file_path, load_img=False, overwrite=False, mend=False, max_gap_size=300, **kwargs):
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
            if hasattr(data, 'outlines'):
                del data['outlines']
            if hasattr(data, 'outlines_list'):
                del data['outlines_list']

    if hasattr(data, 'heights'):
        img=HeightMap(data, name=file_path, **kwargs)
    else:
        img=SegmentedImage(data, name=file_path, **kwargs)
    
    if overwrite:
        img.to_seg_npy(overwrite_img=True)
    return img

def segmentation_from_img(img, name, **kwargs):
    shape=img.shape[:2]
    outlines=np.zeros(shape, dtype=bool)
    masks=np.zeros(shape, dtype=np.uint16)
    data={'name':name,'img':img,'masks':masks,'outlines':outlines}
    img=SegmentedImage(data, **kwargs)
    return img

def segmentation_from_zstack(zstack, name, **kwargs):
    shape=zstack.shape[1:3]
    outlines=np.zeros(shape, dtype=bool)
    masks=np.zeros(shape, dtype=np.uint16)
    data={'name':name,'zstack':zstack,'img':zstack[0],'masks':masks,'outlines':outlines}
    img=HeightMap(data, **kwargs)
    return img

def convert_GUI_seg(seg, multiprocess=False, remove_edge_masks=True, mend=False, max_gap_size=20, export=False, out_path=None):
    ''' convert a segmentation image from the GUI to a format that can be used by the tracking algorithm '''
    from cellpose.utils import masks_to_outlines

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