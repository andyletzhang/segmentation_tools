import numpy as np
from scipy import ndimage
from numba import jit

try:
    import cupy as cp
    from numba import cuda
    from cupyx.scipy import ndimage as cp_ndimage
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("GPU acceleration not available. Falling back to CPU.")

if HAS_GPU:
    @cuda.jit
    def find_peaks_gpu(zstack_derivative, heights, prominence):
        x, y = cuda.grid(2)
        if x < zstack_derivative.shape[1] and y < zstack_derivative.shape[2]:
            last_peak = 0
            for i in range(1, zstack_derivative.shape[0]-1):
                if (zstack_derivative[i, x, y] > zstack_derivative[i-1, x, y] and 
                    zstack_derivative[i, x, y] > zstack_derivative[i+1, x, y]):

                    min_val = zstack_derivative[i, x, y]
                    for j in range(max(0, i-1), min(zstack_derivative.shape[0], i+2)):
                        if zstack_derivative[j, x, y] < min_val:
                            min_val = zstack_derivative[j, x, y]
                    
                    if zstack_derivative[i, x, y] - min_val >= prominence:
                        last_peak = i
            heights[x, y] = last_peak

    def process_zstack_gpu(zstack, prominence=0.004, sigma=6):
        # Move data to GPU
        zstack_gpu = cp.asarray(zstack, dtype=cp.float32)
        zstack_gpu = normalize_gpu(zstack_gpu)
        zstack_gpu = cp_ndimage.gaussian_filter(zstack_gpu, sigma=(0,sigma,sigma))

        # Calculate derivative
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
        # normalize in place
        data -= bounds[0]
        data /= bounds[1] - bounds[0]
        data[data < 0] = 0
        data[data > 1] = 1
        return data

else:
    # Fallback for CPU execution
    @jit(nopython=True)
    def find_peaks_cpu(zstack_derivative, prominence):
        heights = np.zeros((zstack_derivative.shape[1], zstack_derivative.shape[2]), dtype=np.int64)
        for x in range(zstack_derivative.shape[1]):
            for y in range(zstack_derivative.shape[2]):
                last_peak = 0
                for i in range(1, zstack_derivative.shape[0]-1):
                    if (zstack_derivative[i, x, y] > zstack_derivative[i-1, x, y] and 
                        zstack_derivative[i, x, y] > zstack_derivative[i+1, x, y]):

                        min_val = zstack_derivative[i, x, y]
                        for j in range(max(0, i-1), min(zstack_derivative.shape[0], i+2)):
                            if zstack_derivative[j, x, y] < min_val:
                                min_val = zstack_derivative[j, x, y]

                        if zstack_derivative[i, x, y] - min_val >= prominence:
                            last_peak = i
                heights[x, y] = last_peak
        return heights

    def process_zstack_cpu(zstack, prominence=0.004, sigma=6):
        zstack = normalize_cpu(zstack)
        zstack = ndimage.gaussian_filter(zstack, sigma=(0, sigma, sigma))
        derivative = np.gradient(zstack, axis=0)
        return find_peaks_cpu(-derivative, prominence)

    def normalize_cpu(data):
        bounds = np.percentile(data, [1, 99])
        return (data - bounds[0]) / (bounds[1] - bounds[0])

def process_zstack(zstack, prominence=0.004, sigma=6):
    if HAS_GPU:
        return process_zstack_gpu(zstack, prominence, sigma)
    else:
        return process_zstack_cpu(zstack, prominence, sigma)


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
    integer_labels, height_values=relabel_components(heights)
    adjacencies=find_adjacent_components(integer_labels)
    connected_regions=connect_adjacent_components(adjacencies, height_values, integer_labels)

    labels, counts=np.unique(connected_regions, return_counts=True)
    outliers=labels[counts<min_region_size]

    masked=heights.copy()
    masked[np.isin(connected_regions, outliers)]=-1
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

def height_to_3d_mask(heights, max_height=None):
    height, width=heights.shape
    if max_height is None:
        max_height=np.max(heights)

    output=np.zeros((max_height, height, width), dtype=np.uint8)
    for i in range(max_height):
        output[i]=heights>i

    return output

def expand_slice(s, size=1):
    return slice(max(0, s.start-size), s.stop+size)

def get_heights(membrane, min_region_size=1000, peak_prominence=0.004, sigma=6):
    heights=process_zstack(membrane, peak_prominence, sigma=sigma)

    masked_outliers=get_outliers(heights, min_region_size)

    interpolated=interpolate_outliers(masked_outliers)
    return interpolated

def get_coverslip_z(z_profile, scale=1, precision=0.125, prominence=0.01):
    from scipy.interpolate import CubicSpline
    from scipy.signal import find_peaks

    zstack_size=len(z_profile)*scale
    cs_coating=CubicSpline(np.arange(0, zstack_size, scale), z_profile)

    z_fine = np.arange(0, zstack_size, precision)
    intensity_fine = cs_coating(z_fine)

    bottom_slice=z_fine[find_peaks(np.gradient(intensity_fine, z_fine), prominence=prominence)[0][0]]

    return bottom_slice