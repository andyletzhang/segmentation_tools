import numpy as np
from numba import jit
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from skimage.transform import downscale_local_mean

try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    from numba import cuda

    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print('GPU acceleration not available. Falling back to CPU.')

if HAS_GPU:

    @cuda.jit
    def find_peaks_gpu(zstack_derivative, heights, prominence):
        x, y = cuda.grid(2)
        if x < zstack_derivative.shape[1] and y < zstack_derivative.shape[2]:
            last_peak = 0
            for i in range(1, zstack_derivative.shape[0] - 1):
                if (
                    zstack_derivative[i, x, y] > zstack_derivative[i - 1, x, y]
                    and zstack_derivative[i, x, y] > zstack_derivative[i + 1, x, y]
                ):
                    min_val = zstack_derivative[i, x, y]
                    for j in range(max(0, i - 1), min(zstack_derivative.shape[0], i + 2)):
                        if zstack_derivative[j, x, y] < min_val:
                            min_val = zstack_derivative[j, x, y]

                    if zstack_derivative[i, x, y] - min_val >= prominence:
                        last_peak = i
            heights[x, y] = last_peak

    def process_zstack_gpu(zstack, prominence=0.004, sigma=6, z_sigma=0):
        # Move data to GPU
        zstack_gpu = cp.asarray(zstack, dtype=cp.float32)
        zstack_gpu = normalize_gpu(zstack_gpu)
        zstack_gpu = cp_ndimage.gaussian_filter(zstack_gpu, sigma=(z_sigma, sigma, sigma))

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
        find_peaks_gpu[blockspergrid, threadsperblock](-derivative_gpu, heights_gpu, prominence)

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
        
    def resample_zstack_gpu(zstack: np.ndarray, xy_downsample: int, z_upsample: int):
        zstack_cp = cp.array(zstack)

        coords = np.meshgrid(
            np.arange(0, zstack_cp.shape[0], step=1/z_upsample),
            np.arange(0, zstack_cp.shape[1], step=xy_downsample),
            np.arange(0, zstack_cp.shape[2], step=xy_downsample),
            indexing='ij'
        )
        zstack_interp = cp_ndimage.map_coordinates(
            zstack_cp, 
            cp.array(coords), 
            order=3,
            prefilter=True,
            mode='nearest'
        ).get()

        return zstack_interp

# Fallback for CPU execution
@jit(nopython=True)
def find_peaks_cpu(zstack_derivative, prominence):
    heights = np.zeros((zstack_derivative.shape[1], zstack_derivative.shape[2]), dtype=np.int64)
    # iterate over pixels
    for x in range(zstack_derivative.shape[1]):
        for y in range(zstack_derivative.shape[2]):
            last_peak = 0
            for i in range(1, zstack_derivative.shape[0] - 1):  # iterate through z slices (exclude edges which cannot be peaks)
                if (
                    zstack_derivative[i, x, y] > zstack_derivative[i - 1, x, y]
                    and zstack_derivative[i, x, y] > zstack_derivative[i + 1, x, y]
                ):
                    min_val = zstack_derivative[i, x, y]
                    for j in range(
                        max(0, i - 1), min(zstack_derivative.shape[0], i + 2)
                    ):  # check neighboring points for lowest value
                        if zstack_derivative[j, x, y] < min_val:
                            min_val = zstack_derivative[j, x, y]

                    if zstack_derivative[i, x, y] - min_val >= prominence:  # is peak taller than minimum?
                        last_peak = i
            heights[x, y] = last_peak
    return heights


def process_zstack_cpu(zstack, prominence=0.004, sigma=6, z_sigma=0):
    zstack = normalize_cpu(zstack)
    zstack = ndimage.gaussian_filter(zstack, sigma=(z_sigma, sigma, sigma))
    derivative = np.gradient(zstack, axis=0)
    return find_peaks_cpu(-derivative, prominence)


def normalize_cpu(data):
    bounds = np.percentile(data, [1, 99])
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def process_zstack(zstack, prominence=0.004, sigma=6, z_sigma=0):
    if HAS_GPU:
        return process_zstack_gpu(zstack, prominence, sigma, z_sigma=z_sigma)
    else:
        return process_zstack_cpu(zstack, prominence, sigma, z_sigma=z_sigma)


def relabel_components(labeled_grid):
    # Get unique labels in the grid
    label_values = []
    unique_labels = np.unique(labeled_grid)
    relabeled_grid = np.zeros_like(labeled_grid)
    n_relabels = 0
    for label in unique_labels:
        label_mask = labeled_grid == label
        new_labels, n_new_labels = ndimage.label(label_mask)
        new_labels[new_labels > 0] += n_relabels
        n_relabels += n_new_labels
        label_values += [label] * n_new_labels

        relabeled_grid += new_labels
    return relabeled_grid - 1, label_values


# identify adjacent components
@jit(nopython=True)
def find_adjacent_components(components):
    rows, cols = components.shape
    adjacent = set()
    for i in range(rows):
        for j in range(cols):
            current = components[i, j]
            if i > 0:
                left = components[i - 1, j]
                if left != current:
                    if left < current:
                        adjacent.add((left, current))
                    else:
                        adjacent.add((current, left))
            if j > 0:
                up = components[i, j - 1]
                if up != current:
                    if up < current:
                        adjacent.add((up, current))
                    else:
                        adjacent.add((current, up))
    return adjacent


def connect_adjacent_components(adjacencies, heights, labels):
    import networkx as nx

    edgelist = set()
    for adj in adjacencies:
        value = heights[adj[0]]
        adjacent_value = heights[adj[1]]

        if np.abs(adjacent_value - value) == 1:
            edgelist.add(adj)

    G = nx.Graph()
    G.add_nodes_from(range(len(heights)))
    G.add_edges_from(edgelist)

    component_labels = np.empty_like(heights)
    for i, component in enumerate(nx.connected_components(G)):
        for label in component:
            component_labels[label] = i

    connected_regions = component_labels[labels]

    return connected_regions


def get_outliers(heights, min_region_size=1000):
    integer_labels, height_values = relabel_components(heights)
    adjacencies = find_adjacent_components(integer_labels)
    connected_regions = connect_adjacent_components(adjacencies, height_values, integer_labels)

    labels, counts = np.unique(connected_regions, return_counts=True)
    outliers = labels[counts < min_region_size]

    masked = heights.copy()
    masked[np.isin(connected_regions, outliers)] = -1
    return masked


def interpolate_outliers(masked_outliers):
    interpolated = masked_outliers.copy()
    outliers, n_outliers = ndimage.label(masked_outliers == -1)
    bboxes = ndimage.find_objects(outliers)

    for i, (ylim, xlim) in zip(range(1, n_outliers + 1), bboxes):
        ylim = expand_slice(ylim)
        xlim = expand_slice(xlim)
        outlier_mask = outliers[ylim, xlim] == i
        neighbor_mask = ndimage.binary_dilation(outlier_mask) & (~outlier_mask)
        neighbors = masked_outliers[ylim, xlim][neighbor_mask]
        interp_value = np.mean(neighbors).astype(int)
        interpolated[ylim, xlim][outlier_mask] = interp_value
    return interpolated


def height_to_3d_mask(heights, max_height=None):
    height, width = heights.shape
    if max_height is None:
        max_height = np.max(heights)

    output = np.zeros((max_height, height, width), dtype=np.uint8)
    for i in range(max_height):
        output[i] = heights > i

    return output


def expand_slice(s, size=1):
    return slice(max(0, s.start - size), s.stop + size)


def get_heights(membrane, min_region_size=1000, peak_prominence=0.004, sigma=6, z_sigma=0):
    heights = process_zstack(membrane, peak_prominence, sigma=sigma, z_sigma=z_sigma)

    masked_outliers = get_outliers(heights, min_region_size)

    interpolated = interpolate_outliers(masked_outliers)
    return interpolated


def get_coverslip_z(z_profile, scale=1, precision=0.125, prominence=0.01):
    from scipy.interpolate import CubicSpline
    from scipy.signal import find_peaks

    zstack_len = len(z_profile)
    zstack_coords = np.arange(0, zstack_len) * scale
    zstack_size = zstack_len * scale
    cs_coating = CubicSpline(zstack_coords, z_profile)

    z_fine = np.arange(0, zstack_size, precision)
    intensity_fine = cs_coating(z_fine)

    z_gradient = np.gradient(intensity_fine, z_fine)
    scaled_prominence = prominence * (z_gradient.max() - z_gradient.min())
    bottom_slice = z_fine[find_peaks(z_gradient, prominence=scaled_prominence)[0][0]]

    return bottom_slice


def fit_zernike_defocus(data):
    """
    Fit a 2D array (with NaNs allowed) using a 2nd-order Zernike defocus term.
    Returns: center_x, center_y, coeff, offset
    """
    ny, nx = data.shape
    yy, xx = np.indices(data.shape)

    # mask valid points
    mask = ~np.isnan(data)
    x_valid = xx[mask]
    y_valid = yy[mask]
    z_valid = data[mask]

    # set characteristic radius (use half-diagonal of array)
    R = np.hypot(nx, ny) / 2.0

    def residuals(params):
        cx, cy, c, offset = params
        r = np.sqrt((x_valid - cx) ** 2 + (y_valid - cy) ** 2) / R
        basis = 2 * r**2 - 1
        fit = c * basis + offset
        return fit - z_valid

    # initial guess: center in the middle, no defocus
    x0 = [nx / 2, ny / 2, 0.0, np.nanmean(z_valid)]

    result = least_squares(residuals, x0)
    return result.x  # (center_x, center_y, coeff, offset)


def zernike_defocus_surface(shape, cx, cy, coeff, offset):
    yy, xx = np.indices(shape)
    R = np.hypot(*shape) / 2.0
    r_full = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / R
    basis_full = 2 * r_full**2 - 1
    fitted_surface = coeff * basis_full + offset

    return fitted_surface


def get_fitted_surface(data):
    center_x, center_y, coeff, offset = fit_zernike_defocus(data)
    fitted_surface = zernike_defocus_surface(data.shape, center_x, center_y, coeff, offset)
    return fitted_surface


def resample_zstack(zstack: np.ndarray, xy_downsample: int, z_upsample: int):
    if xy_downsample==1 and z_upsample==1:
        return zstack
        
    if HAS_GPU:
        zstack_resampled = resample_zstack_gpu(zstack, xy_downsample, z_upsample)
    else:
        zstack_resampled = resample_zstack_cpu(zstack, xy_downsample, z_upsample)

    return zstack_resampled
    
def resample_zstack_cpu(zstack: np.ndarray, xy_downsample: int=1, z_upsample: int=1):
    zstack_ds = downscale_local_mean(zstack, (1, xy_downsample, xy_downsample)).astype(zstack.dtype)
    zstack_ds_upsampled = np.zeros(
        (zstack_ds.shape[0] * z_upsample, zstack_ds.shape[1], zstack_ds.shape[2]), dtype=zstack_ds.dtype
    )
    for i in range(zstack_ds.shape[1]):
        for j in range(zstack_ds.shape[2]):
            interpolator = CubicSpline(np.arange(0, zstack_ds.shape[0]), zstack_ds[:, i, j], axis=0)
            zstack_ds_upsampled[:, i, j] = interpolator(np.arange(0, len(zstack), step=1 / z_upsample))
    return zstack_ds_upsampled

def fit_zstack_surface(zstack: np.ndarray, xy_downsample: int = 32, z_upsample: int = 8):
    zstack_resampled = resample_zstack(zstack, xy_downsample, z_upsample)
    zstack_blurred_resampled = ndimage.gaussian_filter(zstack_resampled.astype(np.float64), sigma=(z_upsample, 0, 0))
    zstack_gradient = np.gradient(zstack_blurred_resampled, axis=0)
    surface = np.argmin(zstack_gradient, axis=0)
    cx, cy, coeff, offset = fit_zernike_defocus(surface)

    # err = np.std(surface - zernike_defocus_surface(surface.shape, cx, cy, coeff, offset))
    params = np.array([cx * xy_downsample, cy * xy_downsample, coeff / z_upsample, offset / z_upsample])

    fitted_surface = zernike_defocus_surface(zstack.shape[1:], *params)
    return fitted_surface
