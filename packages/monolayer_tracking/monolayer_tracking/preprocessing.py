import numpy as np
from scipy import ndimage
import cellpose.utils as cp_utils
import cv2
import pickle
from multiprocessing import Pool
from functools import partial


def mend_gaps(masks, max_gap_size):
    '''
    cellpose sometimes leaves a few 0-pixels between segmented cells.
    this method finds gaps below the max gap size and fills them using their neighboring cell IDs.
    '''
    from statistics import multimode
    from scipy.signal import convolve2d

    background=ndimage.label(masks==0)[0]
    background_sizes=np.unique(background, return_counts=True)
    gap_IDs=background_sizes[0][background_sizes[1]<max_gap_size] # gaps below maximal spurious size
    
    if len(gap_IDs)!=0: # found at least one gap, proceed to mend (and subsequently overwrite the outlines channel, etc.)
        for gap_ID in gap_IDs:
            gap_pixels=np.array(np.where(background==gap_ID))
            while 0 in masks[gap_pixels[0], gap_pixels[1]]: # still some pixels are empty
                gap_pixels_bbox=np.array([gap_pixels.min(axis=1), gap_pixels.max(axis=1)]).T # bounding box of the gap
                masks_ROI=masks[gap_pixels_bbox[0,0]-1:gap_pixels_bbox[0,1]+2,gap_pixels_bbox[1,0]-1:gap_pixels_bbox[1,1]+2] # region of interest in the masks channel (zeroes the ROI at the top left corner)
                gap_boundary=np.array(np.where((masks_ROI==0)&(convolve2d(masks_ROI!=0,[[0,1,0],[1,0,1],[0,1,0]])[1:-1,1:-1]>0))) # pixels on the edge of the gap: these will be filled with neighbors
                gap_boundary=np.add(gap_boundary.T,np.array([gap_pixels_bbox[0,0]-1,gap_pixels_bbox[1,0]-1])) # transpose back to full image coordinates
                for x,y in gap_boundary:
                    neighbors=masks[[x+1,x,x-1,x],[y,y+1,y,y-1]]
                    fill_value=np.random.choice(multimode(neighbors[neighbors!=0]))
                    masks[x,y]=fill_value
        mended=True
    else:
        mended=False
    return masks, mended

def masks_to_outlines(masks):
        from scipy.sparse import csc_matrix
        boundaries=cp_utils.masks_to_outlines(masks) # have cellpose find new outlines
        mended_outlines=np.zeros(masks.shape, dtype=np.uint16)
        mended_outlines[boundaries]=masks[boundaries] # attach labels to outlines
        return csc_matrix(mended_outlines)

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
        if np.argmin(image.shape)==2: #RGB
            image=normalize_RGB(image, dtype, quantile, **kwargs)
        elif np.argmin(image.shape)==0: # multipage grayscale
            image=np.array([normalize_grayscale(page, dtype, quantile, **kwargs) for page in image])
    else: # grayscale
        image=normalize_grayscale(image, dtype, quantile, **kwargs)

    return image

def normalize_RGB(color_img, dtype='float32', quantile=(0,1), **kwargs):
    ''' normalize each channel of a color image separately '''
    image=color_img.astype(dtype)
    for n, color_channel in enumerate(np.transpose(image, axes=[2,0,1])):
        image[:,:,n]=normalize_grayscale(color_channel, dtype, quantile)
    return image

def normalize_grayscale(image, dtype='float32', quantile=(0,1), mask_zeros=False):
    ''' normalize data by min and max or by some specified quantile '''
    if mask_zeros:
        masked=np.ma.masked_values(image, 0)
        bounds=np.quantile(masked[~masked.mask].data, quantile)
    else:
        bounds=np.quantile(image, quantile)

    if not np.array_equal(bounds,(0,0)):
        image=(image-bounds[0])/(bounds[1]-bounds[0]).astype(dtype)
        image=np.clip(image, a_min=0, a_max=1)
    return image



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

# test
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

# deprecated, I think-----------------------------------
#
#def denoise_FUCCI(img_channel, auto_denoise=True, denoise_cutoff=21, denoise_h=4, norm_range=None):
#    if norm_range is None:
#        norm_range=np.quantile(img_channel, (0.001, 0.999))
#    img_8bit=((img_channel-norm_range[0])*255/(norm_range[1]-norm_range[0]))
#    img_8bit=np.clip(img_8bit, a_min=0, a_max=255).astype('uint8')
#
#    if auto_denoise:
#        noise_metric=[]
#        for h in np.arange(1,denoise_cutoff):
#            denoised=cv2.fastNlMeansDenoising(img_8bit, h=int(h))
#            noise_metric.append(np.abs(denoised-cv2.medianBlur(denoised, ksize=3)).std())
#
#            if len(noise_metric)<4:
#                continue
#            else:
#                try: # point where noise metric begins monotonic descent
#                    hump=np.where(np.diff(noise_metric)>0)[0].max()
#                except ValueError:
#                    hump=0
#                if np.array_equal((np.diff(np.diff(noise_metric[-4:]))>0).astype(int),[0,1]):
#                    if h-4>hump:
#                        break
#        denoise_h=h
#
#    denoised=cv2.fastNlMeansDenoising(img_8bit, h=int(denoise_h))
#    return denoised, denoise_h
#
#def subtract_background(img_channel, gaussian_sigma=250):
#    blur_bg=ndimage.gaussian_filter(img_channel.astype('float'), sigma=gaussian_sigma, truncate=3) # finding the background via large gaussian blur is the time bottleneck
#    img_subtracted=img_channel-blur_bg
#    img_subtracted=np.clip((img_subtracted-img_subtracted.min())*255, a_min=None, a_max=255**2).astype('uint16')
#    return img_subtracted, blur_bg
#
#def normalize_FUCCI_fluorescence(img, masks=None):
#    # NB: this should eventually be done at a stack level rather than image by image? maintain relative fluorescences as best we can.
#    if masks is not None:
#        img_data=img[masks!=0]
#    else:
#        img_data=img.flatten()
#
#    lower_bound=np.exp(np.log(img_data.astype('float64')+1).mean())
#    std=img_data.std()
#
#    return np.clip((img-lower_bound)/std, a_min=0, a_max=None)
