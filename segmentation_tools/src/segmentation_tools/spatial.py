import numpy as np
import warnings
from typing import List, Tuple, Optional, Union, Callable
from dataclasses import dataclass

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    cp=None
    HAS_GPU = False
    warnings.warn('CuPy not found, GPU acceleration will not be available.')


@dataclass
class AutocorrelationResult:
    """Container for autocorrelation computation results."""

    distances: np.ndarray  # Distance values
    correlation: np.ndarray  # Correlation values
    std : np.ndarray = None  # Standard deviation of correlation values
    characteristic_length: Optional[float] = None  # Length where correlation drops below threshold

def spatial_autocorrelation(
    fovs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    min_distance: float = 0,
    max_distance: Optional[float] = None,
    bins: int = 50,
    use_gpu: bool = False,
    progress_callback: Optional[Callable] = None,
    characteristic_threshold: float = 1 / np.e,
    global_mean: bool = False,
) -> AutocorrelationResult:
    """
    Compute spatial autocorrelation for multiple fields of view with automatic GPU/CPU selection.

    Parameters
    ----------
    fovs : List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        List of (x, y, value) tuples for each field of view
    max_distance : float, optional
        Maximum distance for correlation calculation (default: 90th percentile of distances)
    min_distance : float
        Minimum distance for correlation calculation
    bins : int
        Number of distance bins for correlation calculation
    use_gpu : bool
        Whether to use GPU acceleration if available
    progress_callback : Callable, optional
        Function to track progress
    characteristic_threshold : float
        Threshold for characteristic length calculation
    global_mean : bool
        Whether to use global or local mean for normalization

    Returns
    -------
    AutocorrelationResult
        Container with computed distances, correlations, and characteristic length
    """
    # Determine whether to use GPU
    use_gpu = use_gpu and HAS_GPU

    # Use default progress callback if none provided
    if progress_callback is None:

        def progress_callback(x):
            return x

    # Select appropriate backend
    distances, correlations, errors = _compute_autocorrelation(fovs, min_distance, max_distance, bins, progress_callback, global_mean, use_gpu)

    # Calculate characteristic length
    char_length = None
    if len(correlations) > 0:
        below_threshold = np.where(correlations < characteristic_threshold)[0]
        if len(below_threshold) > 0:
            char_length = distances[below_threshold[0]]

    return AutocorrelationResult(distances=distances, correlation=correlations, std=errors, characteristic_length=char_length)


def _compute_autocorrelation(
    fovs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], min_distance: Optional[float], max_distance: Optional[float], bins: int, progress_callback: Callable, global_mean: bool=False, use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPU implementation of spatial autocorrelation calculation."""
    if use_gpu and HAS_GPU:
        xp = cp
        pdist = _pairwise_distance_gpu
    else:
        from scipy.spatial.distance import pdist
        xp = np
        use_gpu = False

    all_dists = []
    all_stat_products = []
    stat_var_total = 0

    all_means=[]
    all_vars=[]
    ns=[]
    for fov in fovs:
        stat=xp.asarray(fov[2])
        all_means.append(xp.mean(stat))
        all_vars.append(xp.var(stat))
        ns.append(len(stat))
    
    all_means=xp.asarray(all_means)
    all_vars=xp.asarray(all_vars)
    ns=xp.asarray(ns)

    if global_mean:
        mean=np.mean(all_means*ns)/np.sum(ns)
        var=(np.sum(all_vars*(ns-1))+np.sum((all_means-mean)**2*ns))/np.sum(ns)

        all_means=np.full(len(fovs),mean)
        all_vars=np.full(len(fovs),var)

    stat_var_total = xp.sum(all_vars)/len(fovs)

    for (x, y, stat), stat_mean in zip(progress_callback(fovs), all_means):
        x, y, stat = xp.asarray(x), xp.asarray(y), xp.asarray(stat)

        points = xp.column_stack((x, y))
        dists = pdist(points)
        stat_centered = stat - stat_mean
        stat_products = stat_centered[:, None] * stat_centered[None, :]

        # Extract upper triangle of stat products
        i, j = xp.triu_indices(stat.shape[0], k=1)
        stat_products = stat_products[i, j]

        all_dists.append(dists)
        all_stat_products.append(stat_products)

    all_dists = xp.concatenate(all_dists)
    all_stat_products = xp.concatenate(all_stat_products)

    # Compute binned mean and standard deviation
    distances, mean_values, std_values = _binned_mean_std(all_dists, all_stat_products, yscale=stat_var_total, min_distance=min_distance, max_distance=max_distance, bins=bins, use_gpu=use_gpu)
    
    return distances, mean_values, std_values

def _binned_mean_std(x, y, yscale: float=1, min_distance: float=0, max_distance: float|None=None, bins:int=50, use_gpu:bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Select computation library
    if use_gpu and HAS_GPU:
        xp = cp
    else:
        xp = np
        use_gpu = False
    
    # Convert inputs to arrays if they aren't already
    x = xp.asarray(x)
    y = xp.asarray(y)
    
    # Early exit if inputs are empty
    if x.size == 0 or y.size == 0:
        bin_centers = xp.linspace(min_distance, max_distance or 1.0, bins)
        empty_result = xp.full(bins, xp.nan)
        return (bin_centers.get() if use_gpu else bin_centers,
                empty_result.get() if use_gpu else empty_result,
                empty_result.get() if use_gpu else empty_result)
    
    # Calculate max_distance if not provided
    if max_distance is None:
        max_distance = float(xp.percentile(x, 90))
    
    # Create bin edges and centers
    bin_edges = xp.linspace(min_distance, max_distance, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Get bin indices, limiting to valid range
    bin_indices = xp.clip(xp.digitize(x, bin_edges) - 1, 0, bins - 1)
    
    # Create masks for valid values (within range and not NaN)
    valid_range_mask = (x >= min_distance) & (x <= max_distance)
    not_nan_mask = ~xp.isnan(y)
    valid_mask = valid_range_mask & not_nan_mask
    
    # Apply masks
    bin_indices_valid = bin_indices[valid_mask]
    y_valid = y[valid_mask]
    
    # Use vectorized operations for binning
    sum_values = xp.bincount(bin_indices_valid, weights=y_valid, minlength=bins)
    sum_squares = xp.bincount(bin_indices_valid, weights=y_valid**2, minlength=bins)
    count_values = xp.bincount(bin_indices_valid, minlength=bins)
    
    # Calculate statistics, handling division by zero
    nonzero_mask = count_values > 0
    mean_values = xp.full(bins, xp.nan)
    stddev_values = xp.full(bins, xp.nan)
    
    # Compute only for bins with data
    if xp.any(nonzero_mask):
        mean_values[nonzero_mask] = sum_values[nonzero_mask] / count_values[nonzero_mask]
        variance = xp.zeros_like(mean_values)
        variance[nonzero_mask] = (sum_squares[nonzero_mask] / count_values[nonzero_mask]) - (mean_values[nonzero_mask]**2)
        # Correct for negative variance due to numerical issues
        variance = xp.maximum(variance, 0)
        stddev_values[nonzero_mask] = xp.sqrt(variance[nonzero_mask])
    
    # Scale outputs
    y_means = mean_values / yscale
    y_stds = stddev_values / (xp.sqrt(xp.maximum(count_values, 1)) * yscale)
    
    # Return results, moving to CPU if needed
    if use_gpu:
        return bin_centers.get(), y_means.get(), y_stds.get()
    else:
        return bin_centers, y_means, y_stds

def _pairwise_distance_gpu(points: Union[np.ndarray, 'cp.ndarray']) -> 'cp.ndarray':
    """Compute pairwise Euclidean distances on GPU."""
    if not isinstance(points, cp.ndarray):
        points = cp.asarray(points)

    # Compute squared L2 norms
    sq_norms = (points * points).sum(axis=1)

    # Compute pairwise distances using matrix multiplication
    dots = cp.dot(points, points.T)
    distances = cp.sqrt(cp.maximum(sq_norms[:, None] + sq_norms[None, :] - 2 * dots, 0))

    # Extract upper triangle
    return distances[cp.triu_indices(distances.shape[0], 1)]