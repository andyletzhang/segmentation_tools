from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy import ndimage
from skimage import io

try:
    import cupy as xp

    gpu = True
except ImportError:
    print('Cupy not found. Using numpy instead.')
    import numpy as xp

    gpu = False


def get_stitched_boundary(membrane, radius=2):
    """get the boundary of a stitched image by identifying zero values (not just the edge of the image because tiles can be staggered)."""
    from scipy.signal import convolve2d

    boundary = convolve2d(membrane == 0, np.ones((2 * radius + 1, 2 * radius + 1)), mode='same') != 0
    boundary[0] = boundary[-1] = boundary[:, 0] = boundary[:, -1] = True

    return boundary


def remove_edge_masks_tile(membrane, masks, radius=2):
    """remove masks that touch the edge of an image, including touching the edge of a tile in a stitched image"""
    boundary = get_stitched_boundary(membrane, radius)
    # remove all masks that touch the edge
    edge_masks = np.unique(masks[boundary])[1:]

    new_masks = masks.copy()
    new_masks[np.isin(new_masks, edge_masks)] = 0
    new_masks = np.unique(new_masks, return_inverse=True)[1].reshape(
        masks.shape
    )  # renumber masks to consecutive integers with edge masks removed
    return new_masks


def read_height_tif(file_path, zero_to_nan=True):
    height_img = io.imread(file_path).astype(bool)  # binary image
    top_surface = np.argmin(height_img, axis=0).astype(float)  # first zero in the height image at each pixel is the top surface
    if zero_to_nan:
        top_surface[top_surface == 0] = np.nan
    return top_surface, height_img


def mend_gaps(masks, max_gap_size):
    """
    cellpose sometimes leaves a few 0-pixels between segmented cells.
    this method finds gaps below the max gap size and fills them using their neighboring cell IDs.
    """
    background = ndimage.label(masks == 0)[0]
    bg_labels, bg_counts = np.unique(background, return_counts=True)
    gap_labels = bg_labels[bg_counts < max_gap_size]  # gaps below maximal spurious size
    mended = len(gap_labels) > 0

    gap_mask = np.isin(background, gap_labels)

    mended_masks = nearest_interpolation(masks, gap_mask)

    return mended_masks, mended


def nearest_interpolation(arr, mask):
    """
    Interpolate values in an array by nearest non-zero values

    Args:
        arr: Input numpy array to draw values from
        mask: Binary mask of values to be interpolated

    Returns:
        Array with masked values replaced by nearest non-zero values
    """
    from scipy.ndimage import distance_transform_edt

    # Create mask of zero values
    if not np.any(mask):
        return arr.copy()

    # Make a copy and set zeros to nan for distance transform
    arr_nan = arr.copy().astype(float)
    arr_nan[mask] = np.nan

    # Initialize the output array
    result = arr.copy()

    # Get indices of nearest non-zero points using distance transform
    indices = distance_transform_edt(mask, return_distances=False, return_indices=True)

    # Use the indices to fill in the zero values
    result[mask] = arr[tuple(indices[:, mask])]

    return result


def gaussian_parallel(imgs, n_processes=8, progress_bar=None, sigma=5, **kwargs):
    p = Pool(processes=n_processes)
    if progress_bar is None:

        def progress_bar(x):
            return x

        progress_kwargs = {}
    else:
        progress_kwargs = {'total': len(imgs), 'desc': 'computing gaussians'}
    out = [
        x
        for x in progress_bar(
            p.imap(partial(ndimage.gaussian_filter, output=np.float32, sigma=sigma, **kwargs), imgs, chunksize=8),
            **progress_kwargs,
        )
    ]

    return out


def get_quantile(img: np.ndarray, q=(1,99), mask_zeros: bool = False):
    """
    Calculate percentiles for an image using CUDA if available.
    If ignore_zeros is True, percentiles are calculated only on non-zero values.
    The last dimension of the image is assumed to be the color channel. If grayscale, the last dimension should be 1.
    """
    # Transfer data to GPU
    img = xp.asarray(img)

    results = []
    for c in range(img.shape[-1]):
        channel = img[..., c]
        
        results.append(quantile_mono(channel, q=q, mask_zeros=mask_zeros))

    # Transfer results back to CPU
    if gpu:
        return xp.asnumpy(xp.array(results))
    else:
        return xp.array(results)
    
def quantile_mono(img, q=(1, 99), mask_zeros: bool = False, mask_nans: bool=True):
    """Calculate percentiles for a single channel"""
    # Transfer data to GPU
    img = xp.asarray(img)

    if img.max()==0:
        return xp.array([0, 1])
    
    mask=np.zeros_like(img, dtype=bool)
    if mask_nans:
        mask|=xp.isnan(img)
    if mask_zeros:
        # Create mask for non-zero values
        mask |= img == 0
            
    # Calculate percentiles only on non-zero values
    img_filtered = img[~mask]
    bounds = xp.percentile(img_filtered, q=q, axis=None)
   
    return bounds


def normalize(image, dtype='float32', percentile=(1, 99), **kwargs):
    """normalize image data (color or grayscale) between 0 and 100 (min max, or a specified percentile)"""
    image=xp.asarray(image).astype(dtype)
    if image.ndim==2:
        image=image[...,xp.newaxis]
    bounds = get_quantile(image, q=percentile, **kwargs)

    for c in range(image.shape[-1]):
        b=bounds[c]
        if b[0]==b[1]:
            image[..., c] = xp.zeros_like(image[..., c])
        else:
            image[..., c] = xp.clip((image[..., c] - b[0]) / (b[1] - b[0]), 0, 1)

    if gpu:
        return xp.asnumpy(image)
    else:
        return image


def get_fluor_threshold(img, size_threshold, noise_score=0.02, percentile=(0.5, 0.95), tolerance=1):
    from scipy.optimize import minimize_scalar

    ret = minimize_scalar(
        nuclear_threshold_loss,
        bounds=np.percentile(img, percentile),
        args=(img, size_threshold, noise_score),
        options={'xatol': tolerance},
    )
    return ret['x']


def nuclear_threshold_loss(threshold, img, size_threshold, noise_score=0.02):
    labels, n_features = ndimage.label(img > threshold)
    n_nuclei = np.sum(np.unique(labels, return_counts=True)[1][1:] > size_threshold)
    return -(n_nuclei - noise_score * n_features)


def fluorescent_percentages(masks, thresholded_img):
    fluor = ndimage.sum_labels(thresholded_img, labels=masks, index=np.unique(masks)[1:])
    areas = ndimage.sum_labels(np.ones(thresholded_img.shape), labels=masks, index=np.unique(masks)[1:])
    return fluor / areas


# used in suspended dataset
def frame_FUCCI(args, percent_threshold=0.15):
    mask, threshold_red, threshold_green, threshold_orange = args
    # threshold_orange=threshold_red&threshold_green

    fluor_percentages = np.stack(
        [
            fluorescent_percentages(mask, threshold_green),
            fluorescent_percentages(mask, threshold_red),
            fluorescent_percentages(mask, threshold_orange),
        ],
        axis=0,
    )
    fluor_nuclei = np.array([*(fluor_percentages[:2] > percent_threshold), fluor_percentages[2] > 1.5 * percent_threshold]).T
    fluor_nuclei[fluor_nuclei[:, 2], :2] = 0  # wherever cells are orange, turn off red and green

    red_or_green = fluor_nuclei[:, 0] & fluor_nuclei[:, 1]
    fluor_nuclei[red_or_green, np.argmin(fluor_percentages[:2, red_or_green], axis=0)] = (
        0  # if not orange, pick red or green based on which has a higher percentage
    )
    cc_stage_number = np.argmax(fluor_nuclei, axis=1) + 1  # G1->1, S->2, G2->3
    cc_stage_number[np.sum(fluor_nuclei, axis=1) == 0] = 0  # no signal->0

    return cc_stage_number


def parallel_frame_FUCCI(args, percent_threshold=0.15, progress_bar=lambda x, **progress_kwargs: x):
    p = Pool(8)
    results = [
        x
        for x in progress_bar(
            p.imap(partial(frame_FUCCI, percent_threshold=percent_threshold), args), total=len(args), desc='Processing frames'
        )
    ]

    return results
