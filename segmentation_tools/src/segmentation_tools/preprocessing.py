import numpy as np
from scipy import ndimage
from multiprocessing import Pool
from functools import partial
from skimage import io

def get_stitched_boundary(membrane, radius=2):
    ''' get the boundary of a stitched image by identifying zero values (not just the edge of the image because tiles can be staggered). '''
    from scipy.signal import convolve2d

    boundary=convolve2d(membrane==0, np.ones((2*radius+1,2*radius+1)), mode='same')!=0
    boundary[0]=boundary[-1]=boundary[:,0]=boundary[:,-1]=True

    return boundary

def remove_edge_masks_tile(membrane, masks, radius=2):
    ''' remove masks that touch the edge of an image, including touching the edge of a tile in a stitched image '''
    boundary=get_stitched_boundary(membrane, radius)
    # remove all masks that touch the edge
    edge_masks=np.unique(masks[boundary])[1:]
    
    new_masks=masks.copy()
    new_masks[np.isin(new_masks, edge_masks)]=0
    new_masks=np.unique(new_masks, return_inverse=True)[1].reshape(masks.shape) # renumber masks to consecutive integers with edge masks removed
    return new_masks

def read_height_tif(file_path, zero_to_nan=True):
    height_img=io.imread(file_path).astype(bool) # binary image
    top_surface=np.argmin(height_img, axis=0).astype(float) # first zero in the height image at each pixel is the top surface
    if zero_to_nan:
        top_surface[top_surface==0]=np.nan
    return top_surface, height_img

def mend_gaps(masks, max_gap_size):
    '''
    cellpose sometimes leaves a few 0-pixels between segmented cells.
    this method finds gaps below the max gap size and fills them using their neighboring cell IDs.
    '''
    # TODO: implement numba version of this function?
    from statistics import multimode
    from scipy.signal import convolve2d
    mended_masks=masks.copy()
    background=ndimage.label(masks==0)[0]
    bg_labels, bg_counts=np.unique(background, return_counts=True)
    gap_labels=bg_labels[bg_counts<max_gap_size] # gaps below maximal spurious size
    
    if len(gap_labels)!=0: # found at least one gap, proceed to mend (and subsequently overwrite the outlines channel, etc.)
        for gap_label in gap_labels:
            gap_pixels=np.array(np.where(background==gap_label))
            while 0 in mended_masks[gap_pixels[0], gap_pixels[1]]: # still some pixels are empty
                gap_pixels_bbox=np.array([gap_pixels.min(axis=1), gap_pixels.max(axis=1)]).T # bounding box of the gap
                masks_ROI=mended_masks[gap_pixels_bbox[0,0]-1:gap_pixels_bbox[0,1]+2,gap_pixels_bbox[1,0]-1:gap_pixels_bbox[1,1]+2] # region of interest in the masks channel (zeroes the ROI at the top left corner)
                gap_boundary=np.array(np.where((masks_ROI==0)&(convolve2d(masks_ROI!=0,[[0,1,0],[1,0,1],[0,1,0]])[1:-1,1:-1]>0))) # pixels on the edge of the gap: these will be filled with neighbors
                gap_boundary=np.add(gap_boundary.T,np.array([gap_pixels_bbox[0,0]-1,gap_pixels_bbox[1,0]-1])) # transpose back to full image coordinates
                for x,y in gap_boundary:
                    neighbors=mended_masks[[x+1,x,x-1,x],[y,y+1,y,y-1]]
                    fill_value=np.random.choice(multimode(neighbors[neighbors!=0]))
                    mended_masks[x,y]=fill_value
        mended=True
    else:
        mended=False
    return mended_masks, mended

def gaussian_parallel(imgs, n_processes=8, progress_bar=None, sigma=5, **kwargs):
    p=Pool(processes=n_processes)
    if progress_bar is None:
        progress_bar=lambda x: x
        progress_kwargs={}
    else:
        progress_kwargs={'total':len(imgs), 'desc':'computing gaussians'}
    out=[x for x in progress_bar(p.imap(partial(ndimage.gaussian_filter,  output=np.float32, sigma=sigma, **kwargs), imgs, chunksize=8), **progress_kwargs)]

    return out

def normalize(image, dtype='float32', quantile=(0.01, 0.99), **kwargs):
    ''' normalize image data (color or grayscale) between 0 and 1 (min max, or a specified quantile)'''
    image=image.astype(dtype)
    if image.ndim==3: # multichannel image: normalize each channel separately
        if np.argmin(image.shape)==2: # RGB
            image=normalize_RGB(image, dtype, quantile, **kwargs)
        elif np.argmin(image.shape)==0: # multipage grayscale
            image=np.array([normalize_grayscale(page, dtype, quantile, **kwargs) for page in image])
    else: # grayscale
        if image.ndim!=2:
            print('Warning: image has 3 or more dimensions and is not RGB. Normalizing in grayscale.')
        image=normalize_grayscale(image, dtype, quantile, **kwargs)

    return image

def normalize_RGB(color_img, dtype='float32', quantile=(0,1), bounds=None, **kwargs):
    ''' normalize each channel of a color image separately '''
    image=color_img.astype(dtype)
    if bounds is None: bounds=[None, None, None]

    for n, color_channel in enumerate(np.transpose(image, axes=[2,0,1])):
        image[:,:,n]=normalize_grayscale(color_channel, dtype, quantile=quantile, bounds=bounds[n], **kwargs)
    return image

def normalize_grayscale(image, dtype='float32', quantile=(0,1), bounds=None, mask_zeros=True):
    ''' normalize data by min and max or by some specified quantile '''
    if bounds is None:
        if mask_zeros:
            masked=np.ma.masked_values(image, 0)
            bounds=np.quantile(masked[~masked.mask].data, quantile)
        else:
            bounds=np.quantile(image, quantile)

    bounds=np.array(bounds, dtype=dtype).flatten() # just flatten it?? very trashy
    image=image.astype(dtype)
    if not np.array_equal(bounds, (0,0)):
        image=(image-bounds[0])/(bounds[1]-bounds[0])
        image=np.clip(image, a_min=0, a_max=1)
    return image

def renumber_masks(masks):
    ''' renumber masks from 1 to n, where n is the number of unique labels in the image '''
    unique_labels=np.unique(masks)
    renumbered_masks = np.searchsorted(unique_labels, masks)
    return renumbered_masks

from scipy.optimize import minimize_scalar
def get_fluor_threshold(img, size_threshold, noise_score=0.02, quantile=(0.5, 0.95), tolerance=1):
    ret=minimize_scalar(nuclear_threshold_loss, bounds=np.quantile(img, quantile), args=(img, size_threshold, noise_score), options={'xatol':tolerance})
    return ret['x']

def nuclear_threshold_loss(threshold, img, size_threshold, noise_score=0.02):
    labels, n_features=ndimage.label(img>threshold)
    n_nuclei=np.sum(np.unique(labels, return_counts=True)[1][1:]>size_threshold)
    return -(n_nuclei-noise_score*n_features)

def fluorescent_percentages(masks, thresholded_img):
    fluor=ndimage.sum_labels(thresholded_img, labels=masks, index=np.unique(masks)[1:])
    areas=ndimage.sum_labels(np.ones(thresholded_img.shape), labels=masks, index=np.unique(masks)[1:])
    return fluor/areas

# used in suspended dataset
def frame_FUCCI(args, percent_threshold=0.15):
    mask, threshold_red, threshold_green, threshold_orange=args
    #threshold_orange=threshold_red&threshold_green

    fluor_percentages=np.stack([fluorescent_percentages(mask, threshold_green),
                                fluorescent_percentages(mask, threshold_red),
                                fluorescent_percentages(mask, threshold_orange)], axis=0)
    fluor_nuclei=np.array([*(fluor_percentages[:2]>percent_threshold), fluor_percentages[2]>1.5*percent_threshold]).T
    fluor_nuclei[fluor_nuclei[:,2], :2]=0 # wherever cells are orange, turn off red and green

    red_or_green=fluor_nuclei[:,0]&fluor_nuclei[:,1]
    fluor_nuclei[red_or_green, np.argmin(fluor_percentages[:2, red_or_green], axis=0)]=0 # if not orange, pick red or green based on which has a higher percentage
    cc_stage_number=np.argmax(fluor_nuclei, axis=1)+1 # G1->1, S->2, G2->3
    cc_stage_number[np.sum(fluor_nuclei, axis=1)==0]=0 # no signal->0

    return cc_stage_number

def parallel_frame_FUCCI(args, percent_threshold=0.15, progress_bar=lambda x, **progress_kwargs: x):
    p=Pool(8)
    results=[x for x in progress_bar(p.imap(partial(frame_FUCCI, percent_threshold=percent_threshold), args), total=len(args), desc='Processing frames')]

    return results