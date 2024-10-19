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

def tiffpage_frame(tif_pages, t=0, z=0):
    img=tif_pages[t,z]
    if img.shape[0]==1:
        img=img[0].asarray()
    elif img.shape[0]>1:
        img=np.array([a.asarray() for a in img])
        if img.shape[0]!=3:
            print(f'Warning: unexpected number of color channels {img.shape[0]}')
        img=img.transpose(1,2,0)
    return img

def tiffpage_zstack(tif_pages, t=0):
    zstack=tif_pages[t]
    if zstack.shape[1]==1:
        return np.array([a.asarray() for a in zstack[:,0]]) # mono z stack
    else:
        shape=zstack.shape
        zstack=np.array([a.asarray() for a in zstack.flatten()]).reshape(shape).transpose(0,2,3,1)
        if zstack.shape[-1]!=3:
            print(f'Warning: unexpected number of color channels {zstack.shape[-1]}')
        return zstack
