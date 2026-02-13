import gc
import numpy as np
from numba import jit
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares, curve_fit
from skimage.transform import downscale_local_mean

try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    from numba import cuda
    import torch

    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print('GPU acceleration not available. Falling back to CPU.')

if HAS_GPU:
    def deep_clean_gpu():
        # 1. Release Python object references (Crucial!)
        # If a Python variable still points to a GPU array, the VRAM cannot be freed.
        gc.collect()
        
        # 2. Clear PyTorch Cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # 3. Clear CuPy Memory Pools
        if cp.cuda.is_available():
            pool = cp.get_default_memory_pool()
            pool.free_all_blocks()
            
            pinned_pool = cp.get_default_pinned_memory_pool()
            pinned_pool.free_all_blocks()
            
            # 4. Clear CuPy Kernel/FFT Caches (The hidden creeper)
            # Repeatedly compiling kernels for slightly different array sizes can bloat memory.
            cp.fft.config.get_plan_cache().clear()
            
            # Clear the memoization cache for compiled kernels
            # (Note: This is an internal CuPy registry that can grow over time)
            try:
                cp._default_memory_pool.free_all_blocks()
            except:
                pass
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

    def get_optimal_tile_size(nz, byte_size=4, safety_factor=0.5):
        """Calculates max XY square tile size that fits in VRAM given the Z-depth."""
        free_mem, _ = cp.cuda.runtime.memGetInfo()
        
        # Memory footprint per pixel column:
        # 1. Input Tile (float32)
        # 2. Normalized Tile (in-place or copy, let's assume copy for safety)
        # 3. Gaussian Filter Output (float32)
        # 4. Gradient Output (float32)
        # 5. Heights Output (int64 -> 8 bytes, only 1 per XY pixel, negligible Z factor)
        
        # Bytes per voxel * Z depth
        bytes_per_voxel_column = (4 * 4) * nz # 4 arrays of float32
        
        usable_mem = free_mem * safety_factor
        pixels_per_tile = usable_mem // bytes_per_voxel_column
        
        tile_side = int(np.sqrt(pixels_per_tile))
        # Clamp to reasonable powers of 2 for GPU efficiency
        return min(max(tile_side, 64), 2048)

    def process_zstack_gpu(zstack, prominence=0.004, sigma=6, z_sigma=0):
        """
        Processes the Z-stack in overlapping XY tiles to prevent OOM.
        """
        nz, ny, nx = zstack.shape
        
        # 1. Global Normalization Statistics (CPU)
        # We must compute this globally first, otherwise each tile will be 
        # normalized differently, creating "seams" in the final image.
        # Use a strided sample for speed if zstack is huge
        sample = zstack[::max(1, nz//50), ::max(1, ny//50), ::max(1, nx//50)]
        bounds = np.percentile(sample, [1, 99])
        lower_b, upper_b = bounds[0], bounds[1]
        norm_range = upper_b - lower_b
        if norm_range == 0:
            norm_range = 1.0

        # 2. Determine Tile Size
        tile_size = get_optimal_tile_size(nz)
        
        # Output array (CPU)
        heights_final = np.zeros((ny, nx), dtype=np.int64)

        # Padding amount: 4 * sigma is usually sufficient for Gaussian to decay to zero
        pad = int(4 * sigma)
        
        # 3. Iterate over XY tiles
        for y_start in range(0, ny, tile_size):
            for x_start in range(0, nx, tile_size):
                y_end = min(y_start + tile_size, ny)
                x_end = min(x_start + tile_size, nx)
                
                # Define the "Valid" region (where we want results)
                h_valid = y_end - y_start
                w_valid = x_end - x_start
                
                # Define the "Padded" region (what we load to GPU)
                y_pad_start = max(0, y_start - pad)
                y_pad_end = min(ny, y_end + pad)
                x_pad_start = max(0, x_start - pad)
                x_pad_end = min(nx, x_end + pad)
                
                # Load Tile to GPU
                # We copy the slice to avoid sending the whole array
                tile_cpu = zstack[:, y_pad_start:y_pad_end, x_pad_start:x_pad_end]
                zstack_gpu = cp.array(tile_cpu, dtype=cp.float32)
                
                # --- Processing Pipeline (Same as original) ---
                
                # A. Normalize using GLOBAL bounds (In-place)
                zstack_gpu -= lower_b
                zstack_gpu /= norm_range
                cp.clip(zstack_gpu, 0, 1, out=zstack_gpu)
                
                # B. Gaussian Filter
                # Note: We filter the *entire* padded tile
                zstack_gpu = cp_ndimage.gaussian_filter(zstack_gpu, sigma=(z_sigma, sigma, sigma))
                
                # C. Gradient
                derivative_gpu = cp.gradient(zstack_gpu, axis=0)
                
                # D. Find Peaks
                # Prepare output for this tile
                tile_h, tile_w = derivative_gpu.shape[1], derivative_gpu.shape[2]
                heights_gpu = cp.zeros((tile_h, tile_w), dtype=cp.int64)
                
                threadsperblock = (16, 16)
                blockspergrid_x = (tile_h + threadsperblock[0] - 1) // threadsperblock[0]
                blockspergrid_y = (tile_w + threadsperblock[1] - 1) // threadsperblock[1]
                blockspergrid = (blockspergrid_x, blockspergrid_y)
                
                # Pass NEGATIVE derivative as per original logic
                find_peaks_gpu[blockspergrid, threadsperblock](-derivative_gpu, heights_gpu, prominence)
                
                # --- Crop and Store ---
                
                # We need to extract the valid center region from the padded result
                # Calculate offsets relative to the padded tile
                rel_y_start = y_start - y_pad_start
                rel_x_start = x_start - x_pad_start
                
                # Crop the GPU array
                valid_heights_gpu = heights_gpu[
                    rel_y_start : rel_y_start + h_valid, 
                    rel_x_start : rel_x_start + w_valid
                ]
                
                # Copy to CPU and place in final image
                heights_final[y_start:y_end, x_start:x_end] = valid_heights_gpu.get()

        deep_clean_gpu()  # Ensure all GPU memory is freed after processing
        return heights_final

    # def process_zstack_gpu(zstack, prominence=0.004, sigma=6, z_sigma=0):
    #     # Move data to GPU
    #     zstack_gpu = cp.asarray(zstack, dtype=cp.float32)
    #     zstack_gpu = normalize_gpu(zstack_gpu)
    #     zstack_gpu = cp_ndimage.gaussian_filter(zstack_gpu, sigma=(z_sigma, sigma, sigma))

    #     # Calculate derivative
    #     derivative_gpu = cp.gradient(zstack_gpu, axis=0)

    #     # Prepare output array
    #     heights_gpu = cp.zeros((derivative_gpu.shape[1], derivative_gpu.shape[2]), dtype=cp.int64)

    #     # Set up grid for CUDA kernel
    #     threadsperblock = (16, 16)
    #     blockspergrid_x = (derivative_gpu.shape[1] + threadsperblock[0] - 1) // threadsperblock[0]
    #     blockspergrid_y = (derivative_gpu.shape[2] + threadsperblock[1] - 1) // threadsperblock[1]
    #     blockspergrid = (blockspergrid_x, blockspergrid_y)

    #     # Run kernel
    #     find_peaks_gpu[blockspergrid, threadsperblock](-derivative_gpu, heights_gpu, prominence)

    #     # Move result back to CPU and return
    #     return cp.asnumpy(heights_gpu)

    # def normalize_gpu(data):
    #     bounds = cp.percentile(data, [1, 99])
    #     # normalize in place
    #     data -= bounds[0]
    #     data /= bounds[1] - bounds[0]
    #     data[data < 0] = 0
    #     data[data > 1] = 1
    #     return data
        
    def get_optimal_z_chunk(ny, nx, xy_downsample, dtype_size=4, safety_factor=0.5):
        """
        Calculates how many Z-planes can fit in VRAM at once.
        """
        # Get current GPU memory status (free, total)
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        
        # Target XY dimensions after downsampling
        out_ny = int(np.ceil(ny / xy_downsample))
        out_nx = int(np.ceil(nx / xy_downsample))
        pixels_per_plane = out_ny * out_nx

        # Memory cost per output Z-plane:
        # 1. The Output Array Chunk (on GPU): 1 * pixels_per_plane * dtype_size
        # 2. The Coordinate Grid (3 arrays): 3 * pixels_per_plane * 4 bytes (coords are always float32)
        # 3. The Input Slice (Approximate): (1 / z_upsample) * pixels_per_plane * dtype_size * (xy_downsample^2)
        #    (Note: Input slice cost is tricky because of the 'halo', so we estimate conservatively)
        
        cost_per_plane = (
            (1 * pixels_per_plane * dtype_size) +       # Output Image Storage
            (3 * pixels_per_plane * 4) +                # Coordinate Grids (Biggest consumer)
            (pixels_per_plane * dtype_size * 2)         # Overhead (Input slice copy + CUDA context)
        )

        # Calculate max planes
        usable_mem = free_mem * safety_factor
        max_z = int(usable_mem // cost_per_plane)
        
        # Clamp to reasonable limits (minimum 1, max 2048 to prevent timeouts)
        return max(1, min(max_z, 2048))

    def resample_zstack_gpu(zstack: np.ndarray, xy_downsample: int, z_upsample: int):
        nz, ny, nx = zstack.shape
        
        # --- Auto-Detect Chunk Size ---
        z_chunk_size = get_optimal_z_chunk(ny, nx, xy_downsample, zstack.itemsize)
        # ------------------------------

        # 1. Setup Global Coordinates (CPU)
        z_targets_global = np.arange(0, nz, step=1/z_upsample)
        y_range = cp.arange(0, ny, step=xy_downsample, dtype=cp.float32)
        x_range = cp.arange(0, nx, step=xy_downsample, dtype=cp.float32)
        
        out_shape = (len(z_targets_global), len(y_range), len(x_range))
        zstack_interp = np.empty(out_shape, dtype=zstack.dtype)
        
        # 2. Process Chunks
        SPLINE_PADDING = 12  # Slightly increased for safety
        
        for z_start in range(0, out_shape[0], z_chunk_size):
            z_end = min(z_start + z_chunk_size, out_shape[0])
            z_coords_chunk = z_targets_global[z_start:z_end]
            
            # Calculate Input ROI with Padding
            z_min_input = int(np.floor(z_coords_chunk.min())) - SPLINE_PADDING
            z_max_input = int(np.ceil(z_coords_chunk.max())) + SPLINE_PADDING
            
            # Clamp and Load
            z_min_clamped = max(0, z_min_input)
            z_max_clamped = min(nz, z_max_input)
            input_chunk_gpu = cp.array(zstack[z_min_clamped:z_max_clamped])
            
            # Adjust Z coordinates relative to the loaded chunk
            z_coords_gpu = cp.array(z_coords_chunk - z_min_clamped, dtype=cp.float32)
            
            # Meshgrid & Interpolate
            z_grid, y_grid, x_grid = cp.meshgrid(z_coords_gpu, y_range, x_range, indexing='ij')
            coords_gpu = cp.stack([z_grid, y_grid, x_grid], axis=0)
            
            out_chunk = cp_ndimage.map_coordinates(
                input_chunk_gpu, 
                coords_gpu, 
                order=3,
                prefilter=True,
                mode='nearest' 
            ).get()
            
            zstack_interp[z_start:z_end] = out_chunk

        deep_clean_gpu()
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

def tilted_sphere(xy_tuple, A, x0, y0, B, C, D):
    x, y = xy_tuple
    # Paraboloid term: A*(x-x0)^2 + A*(y-y0)^2
    sphere_term = A * ((x - x0)**2 + (y - y0)**2)
    # Plane term: B*x + C*y + D
    plane_term = B * x + C * y + D
    return sphere_term + plane_term

def fit_tilted_sphere(surface):
    # initial guess parameters
    p0 = [
        -0.01,           # A: Initial curvature guess (gentle bowl)
        surface.shape[1] / 2,         # x0: Center X
        surface.shape[0] / 2,         # y0: Center Y
        0,              # B: Slope X (no tilt)
        0,              # C: Slope Y
        np.mean(surface) # D: Offset
    ]

    x = np.arange(surface.shape[1])
    y = np.arange(surface.shape[0])
    X, Y = np.meshgrid(x, y)

    popt, pcov = curve_fit(tilted_sphere, (X.ravel(), Y.ravel()), surface.ravel(), p0=p0)
    return popt

def make_tilt_sphere_surface(shape, curvature, center_x, center_y, slope_x, slope_y, offset):
    yy, xx = np.indices(shape)
    surface = (curvature * ((xx - center_x)**2 + (yy - center_y)**2) +
               slope_x * xx + slope_y * yy + offset)

    return surface

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

def fit_zstack_zernike_surface(zstack: np.ndarray, xy_downsample: int = 32, z_upsample: int = 8, z_scale=1, xy_scale=0.325):
    zstack_resampled = resample_zstack(zstack, xy_downsample, z_upsample)
    sigma=42/xy_downsample/xy_scale
    z_sigma=z_upsample/z_scale
    zstack_blurred_resampled = ndimage.gaussian_filter(zstack_resampled.astype(np.float64), sigma=(z_sigma, sigma, sigma))
    zstack_gradient = np.gradient(zstack_blurred_resampled, axis=0)
    surface = np.argmin(zstack_gradient, axis=0)
    cx, cy, coeff, offset = fit_zernike_defocus(surface)

    # err = np.std(surface - zernike_defocus_surface(surface.shape, cx, cy, coeff, offset))
    params = np.array([cx * xy_downsample, cy * xy_downsample, coeff / z_upsample, offset / z_upsample])

    fitted_surface = zernike_defocus_surface(zstack.shape[1:], *params)
    return fitted_surface

def fit_zstack_surface(zstack: np.ndarray, xy_downsample: int = 16, z_upsample: int = 8, z_scale=1, xy_scale=0.65, surface_upscale: int = 1):
    zstack_resampled = resample_zstack(zstack, xy_downsample, z_upsample)
    sigma=42/xy_downsample/xy_scale
    z_sigma=z_upsample/z_scale
    zstack_blurred_resampled = ndimage.gaussian_filter(zstack_resampled.astype(np.float64), sigma=(z_sigma, sigma, sigma))
    zstack_gradient = np.gradient(zstack_blurred_resampled, axis=0)
    surface = np.argmin(zstack_gradient, axis=0)
    curvature, cx, cy, mx, my, b = fit_tilted_sphere(surface)
    params = np.array([curvature / z_upsample / (xy_downsample*surface_upscale)**2, cx * xy_downsample * surface_upscale, cy * xy_downsample * surface_upscale, mx / z_upsample / (xy_downsample*surface_upscale), my / z_upsample / (xy_downsample*surface_upscale), b / z_upsample])
    shape = (zstack.shape[1]*surface_upscale, zstack.shape[2]*surface_upscale) if surface_upscale>1 else zstack.shape[1:3]
    fitted_surface = make_tilt_sphere_surface(shape, *params)

    return fitted_surface
