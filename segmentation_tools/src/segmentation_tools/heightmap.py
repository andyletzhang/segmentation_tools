import numpy as np
from scipy import ndimage
from cupyx.scipy import ndimage as cp_ndimage
from numba import jit, prange
import time

try:
    import cupy as cp
    from numba import cuda
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("GPU acceleration not available. Install CuPy and ensure CUDA is properly set up.")

@cuda.jit
def find_peaks_gpu(derivative2, heights, prominence):
    x, y = cuda.grid(2)
    if x < derivative2.shape[1] and y < derivative2.shape[2]:
        last_peak = 0
        for i in range(1, derivative2.shape[0]-1):
            if (derivative2[i, x, y] > derivative2[i-1, x, y] and 
                derivative2[i, x, y] > derivative2[i+1, x, y]):
                
                # Check prominence
                min_val = derivative2[i, x, y]
                for j in range(max(0, i-1), min(derivative2.shape[0], i+2)):
                    if derivative2[j, x, y] < min_val:
                        min_val = derivative2[j, x, y]
                
                if derivative2[i, x, y] - min_val >= prominence:
                    last_peak = i
        
        heights[x, y] = last_peak

def process_zstack_gpu(zstack, prominence=0.004):
    # Move data to GPU
    zstack_gpu = cp.asarray(zstack)
    zstack_gpu = normalize_gpu(zstack_gpu)
    zstack_gpu = cp_ndimage.gaussian_filter(zstack_gpu, sigma=(0,6,6))
    # Calculate second derivative
    derivative_gpu = cp.gradient(zstack_gpu, axis=0)
    
    # Prepare output array
    heights_gpu = cp.zeros((derivative_gpu.shape[1], derivative_gpu.shape[2]), dtype=cp.int64)
    
    # Set up grid for CUDA kernel
    threadsperblock = (16, 16)
    blockspergrid_x = (derivative_gpu.shape[1] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (derivative_gpu.shape[2] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # Run kernel
    find_peaks_gpu[blockspergrid, threadsperblock](
        -derivative_gpu, heights_gpu, prominence)
    
    # Move result back to CPU and return
    return cp.asnumpy(heights_gpu)

def normalize_gpu(data):
    bounds = cp.percentile(data, [1, 99])
    return (data - bounds[0]) / (bounds[1] - bounds[0])

def gaussian_filter_gpu(data, sigma):
    return cp.asnumpy(cp.ndimage.gaussian_filter(cp.asarray(data), sigma))

def relabel_components(labeled_grid):
    # Get unique labels in the grid
    label_values=[]
    unique_labels = np.unique(labeled_grid)
    relabeled_grid=np.zeros_like(labeled_grid)
    n_relabels=0
    for label in unique_labels:
        label_mask=(labeled_grid==label)
        new_labels, n_new_labels=ndimage.label(label_mask)
        new_labels[new_labels>0]+=n_relabels
        n_relabels+=n_new_labels
        label_values+=[label]*n_new_labels
        
        relabeled_grid+=new_labels
    return relabeled_grid-1, label_values

# identify adjacent components
@jit(nopython=True)
def find_adjacent_components(components):
    rows, cols = components.shape
    adjacent = set()
    for i in range(rows):
        for j in range(cols):
            current=components[i,j]
            if i>0:
                left=components[i-1,j]
                if left!=current:
                    if left<current:
                        adjacent.add((left, current))
                    else:
                        adjacent.add((current, left))
            if j>0:
                up=components[i,j-1]
                if up!=current:
                    if up<current:
                        adjacent.add((up, current))
                    else:
                        adjacent.add((current, up))
    return adjacent

def connect_adjacent_components(adjacencies, heights, labels):
    import networkx as nx
    edgelist=set()
    for adj in adjacencies:
        value=heights[adj[0]]
        adjacent_value=heights[adj[1]]

        if np.abs(adjacent_value-value)==1:
            edgelist.add(adj)

    G=nx.Graph()
    G.add_nodes_from(range(len(heights)))
    G.add_edges_from(edgelist)

    component_labels=np.empty_like(heights)
    for i, component in enumerate(nx.connected_components(G)):
        for label in component:
            component_labels[label]=i

    connected_regions=component_labels[labels]

    return connected_regions

def get_outliers(heights, min_region_size=1000):
    start=time.time()
    print("Relabeling components...")
    integer_labels, height_values=relabel_components(heights)
    print(f"Relabeling took {time.time()-start:.2f} seconds.")

    start=time.time()
    print("Finding adjacent components...")
    adjacencies=find_adjacent_components(integer_labels)
    print(f"Adjacent component search took {time.time()-start:.2f} seconds.")

    start=time.time()
    print("Connecting adjacent components...")
    connected_regions=connect_adjacent_components(adjacencies, height_values, integer_labels)
    print(f"Connected component search took {time.time()-start:.2f} seconds.")

    start=time.time()
    print("Masking outliers...")
    labels, counts=np.unique(connected_regions, return_counts=True)
    outliers=labels[counts<min_region_size]

    masked=heights.copy()
    masked[np.isin(connected_regions, outliers)]=-1
    print(f"Outlier masking took {time.time()-start:.2f} seconds.")
    return masked

def interpolate_outliers(masked_outliers):
    interpolated=masked_outliers.copy()
    outliers, n_outliers=ndimage.label(masked_outliers==-1)
    bboxes=ndimage.find_objects(outliers)

    for i, (ylim, xlim) in zip(range(1, n_outliers+1), bboxes):
        ylim=expand_slice(ylim)
        xlim=expand_slice(xlim)
        outlier_mask=(outliers[ylim, xlim]==i)
        neighbor_mask=ndimage.binary_dilation(outlier_mask)&(~outlier_mask)
        neighbors=masked_outliers[ylim, xlim][neighbor_mask]
        interp_value=np.mean(neighbors).astype(int)
        interpolated[ylim, xlim][outlier_mask]=interp_value
    return interpolated

def expand_slice(s, size=1):
    return slice(max(0, s.start-size), s.stop+size)

def get_heights(membrane, min_region_size=1000, peak_prominence=0.004):
    start=time.time()
    print("Computing heights...")
    heights=process_zstack_gpu(membrane, peak_prominence)
    print(f"Height computation took {time.time()-start:.2f} seconds.")

    masked_outliers=get_outliers(heights, min_region_size)

    start=time.time()
    print("Interpolating outliers...")
    interpolated=interpolate_outliers(masked_outliers)
    print(f"Outlier interpolation took {time.time()-start:.2f} seconds.")
    return interpolated

def height_to_3d_mask(heights, max_height=None):
    height, width=heights.shape
    if max_height is None:
        max_height=np.max(heights)

    output=np.zeros((max_height, height, width), dtype=np.uint8)
    for i in range(max_height):
        output[i]=heights>i

    return output