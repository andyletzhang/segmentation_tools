import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def outlines_list(masks):
    """Get outlines of masks as a list to loop over for plotting.

    Args:
        masks (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)

    Returns:
        list: List of outlines as pixel coordinates.

    """
    import cv2
    from scipy.ndimage import find_objects

    outpix = []
    boundaries = find_objects(masks)
    for (y,x), n in zip(boundaries, np.unique(masks)[1:]):
        mn=masks[y,x]==n
        if np.any(mn):
            contours, _ = cv2.findContours(mn.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cmax = np.argmax([c.shape[0] for c in contours])
            pix = contours[cmax].astype(int).squeeze()
            if len(pix) > 4:
                outpix.append(pix+np.array([x.start, y.start]))
            else:
                outpix.append(np.zeros((0, 2)))

    return outpix

def masks_to_outlines(masks):
    """Get outlines of masks as a binary image.

    Args:
        masks (ndarray): masks (0=no cells, 1=first cell, 2=second cell,...)

    Returns:
        ndarray: Binary image with outlines of masks.

    """
    outpix = outlines_list(masks)
    out = np.zeros(masks.shape, dtype=bool)
    for i, pix in enumerate(outpix):
        x,y=pix.T
        out[y,x] = True
    return out

def masks_to_rois(masks):
    from roifile import ImagejRoi
    outlines = outlines_list(masks)

    rois=[]
    for outline in outlines:
        if len(outline) > 0:
            rois.append(ImagejRoi.frompoints(outline))
        else:
            rois.append(None)

    return rois

def picklesize(o):
    '''
    Measures approximate RAM usage (in MB) by how much space it would take to save using pickle. Seems pretty reliable for my purposes.
    '''
    return len(pickle.dumps(o))/(1024**2)


def cell_scalar_attrs(cell):
    ''' Return all common attributes which are scalar cell attributes '''
    attrs=set(dir(cell))
    ignored_attrs={'vertex_area','shape_parameter','sorted_vertices','vertex_perimeter'}
    test_attrs=attrs-ignored_attrs
    # Collect attributes to remove instead of modifying the set in place
    to_remove = set()
    for attr in test_attrs:
        if attr.startswith('_'):
            to_remove.add(attr)
            continue
        try:
            val = getattr(cell, attr)
        except AttributeError:
            to_remove.add(attr)
            continue
        if not np.isscalar(val):
            to_remove.add(attr)

    # Remove collected attributes from test_attrs
    test_attrs -= to_remove
    return test_attrs

def rotate_points(points:np.ndarray, shape, k=1):
    M, N = shape  # Original shape (rows, cols)
    
    if k % 4 == 1:  # 90° counterclockwise
        points=[(y, M - 1 - x) for x, y in points]
    elif k % 4 == 2:  # 180° counterclockwise
        points=[(M - 1 - x, N - 1 - y) for x, y in points]
    elif k % 4 == 3:  # 270° counterclockwise
        points=[(N - 1 - y, x) for x, y in points]
    else:
        pass

    return np.array(points)

def circular_colony_tiles(input, output=None, use_colonies='all', x_spacing=290, y_spacing=410, padding=400, verbose=True):
    """
    This script is designed to tile microscope stage positions for the MetaMorph imaging software.

    Parameters:
        input (str): The filename of the input file containing stage positions. The file is assumed to be an .STG (raw text file) exported by MetaMorph. 
        output (str, optional): The filename for the output file containing the tiled stage positions, also a .STG path. If not provided, a default name is generated based on the input filename.
        x_spacing (int, optional): The spacing between tiles in the x-direction. Default is 290.
        y_spacing (int, optional): The spacing between tiles in the y-direction. Default is 410.
        padding (int, optional): The padding added around each colony when calculating the tiling scheme. Default is 400.

    Returns:
        None: The function writes the tiled stage positions to a .STG file.
    """
    
    if not output:
        output=input.replace('.STG', '_tiled.STG')

    head, points = read_STG(input)

    colony_bounds=np.split(points, len(points)/4) # Split points into colonies. Assuming every 4 stage positions represent a colony (left right up down, in any order).

    if use_colonies!='all':
        colony_bounds=[colony_bounds[i] for i in use_colonies]

    tiled=[]
    for n, colony_bound in enumerate(colony_bounds):
        z_position=colony_bound['current_z'].mean() # mean z position of colony is used as the z for each tile
        # get colony bounding box
        max_corner=colony_bound[['x','y']].max()
        min_corner=colony_bound[['x','y']].min()

        tile_centers=grid_from_bbox(min_corner, max_corner, x_spacing, y_spacing, padding=padding)

        # Assign z positions, etc. to each stage position
        tile_centers[['current_z', 'starting_z']]=z_position
        tile_centers['position']=f'Colony{n+1}'
        tile_centers['stage_number']=[f'_{i}' for i in range(1, len(tile_centers)+1)]
        tile_centers[['AF_offset','FALSE','-9999', 'TRUE_1', 'TRUE_2', '0', '-1', 'endline']]=np.array(colony_bound[['AF_offset', 'FALSE', '-9999', 'TRUE_1', 'TRUE_2', '0', '-1', 'endline']])[0]
        
        tiled.append(tile_centers)

    tiled=pd.concat(tiled)

    if verbose: # some visualizations that the tiling worked
        for n, colony in tiled.groupby('position'):
            plt.scatter(colony['x'], colony['y'], label=n) # scatterplot
            print(colony['position'].iloc[0]+f': {colony.x.nunique()} columns, {colony.y.nunique()} rows') # number of rows and columns for each colony
        
        print(tiled.groupby('position').size()) # number of total tiles for each colony

        plt.scatter(points['x'], points['y'], c='r') # scatter inputted stage points
        plt.gca().set_aspect('equal')

    tiled['position']=tiled['position']+tiled['stage_number'] # unique name for each stage position
    tiled=tiled[points.columns] # reorder columns for exporting

    # write to file
    head[-1]=f'{len(tiled)}\n'
    write_STG(output, head, tiled)

def grid_from_bbox(min_corner, max_corner, x_spacing, y_spacing, padding=0, pad_from_center=True, reorder=True):
    from math import ceil

    grid_size=max_corner-min_corner+2*padding
    
    # number of tiles needed in x and y directions
    n_x_tiles=ceil(grid_size['x']/x_spacing)
    n_y_tiles=ceil(grid_size['y']/y_spacing)


    # Generate x and y coordinates of tiles
    if pad_from_center:
        center=(max_corner+min_corner)/2
        starting_point=center-np.array([n_x_tiles*x_spacing/2, n_y_tiles*y_spacing/2])
    else:
        starting_point=min_corner-padding

    x_tiles=np.arange(1,n_x_tiles)*x_spacing+starting_point['x']
    y_tiles=np.arange(1,n_y_tiles)*y_spacing+starting_point['y']

    # DataFrame of tile centers
    tile_centers=np.stack(np.meshgrid(x_tiles, y_tiles), axis=-1)
    if reorder:
        tile_centers[1::2,:]=tile_centers[1::2,::-1] # reverse every other row

    tile_centers=pd.DataFrame(tile_centers.reshape(-1,2), columns=['x','y'])

    return tile_centers


def read_STG(filename):
    """
    Reads a MetaMorph .STG file and returns a DataFrame of the stage positions.

    Parameters:
        filename (str): The filename of the .STG file to be read.

    Returns:
        DataFrame: A DataFrame containing the stage positions.
    """
    file=open(filename, 'r')
    head = [next(file) for _ in range(4)] # Read header of input file

    # Read stage positions into DataFrame
    points=pd.DataFrame([line.split(', ') for line in file], columns=['position', 'x','y','current_z', 'AF_offset','starting_z','FALSE','-9999', 'TRUE_1', 'TRUE_2', '0', '-1', 'endline'])
    points=points.astype({'x': 'int32', 'y': 'int32', 'current_z': 'float64', 'AF_offset': 'int32', 'starting_z': 'float64'})

    return head, points


def write_STG(filename, head, data):
    """
    Writes a DataFrame of stage positions to a MetaMorph .STG file.

    Parameters:
        filename (str): The filename of the .STG file to be written.
        points (DataFrame): A DataFrame containing the stage positions to be written.

    Returns:
        None: The function writes the stage positions to a .STG file.
    """
    with open(filename, 'w') as file:
        file.writelines(head)
        for _, row in data.iterrows():
            file.write(f'"{row["position"]}", {int(row["x"])}, {int(row["y"])}, {row["current_z"]:.2f}, {row["AF_offset"]}, {row["starting_z"]:.2f}, {row["FALSE"]}, {row["-9999"]}, {row["TRUE_1"]}, {row["TRUE_2"]}, {row["0"]}, {row["-1"]}, {row["endline"]}')