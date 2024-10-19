from . import preprocessing
from cellpose import utils, models
from skimage import io
from tqdm.notebook import tqdm
import numpy as np
from pathlib import Path

def get_stitched_boundary(membrane, radius=2):
    from scipy.signal import convolve2d

    boundary=convolve2d(membrane==0, np.ones((2*radius+1,2*radius+1)), mode='same')!=0 # find pixels that are adjacent to zeros
    boundary[0]=boundary[-1]=boundary[:,0]=boundary[:,-1]=True # edge pixels are also considered boundary

    return boundary

def remove_edge_masks(membrane, masks, radius=2):
    boundary=get_stitched_boundary(membrane, radius)
    # remove all masks that touch the edge
    edge_masks=np.unique(masks[boundary])[1:]

    new_masks=masks.copy()
    new_masks[np.isin(new_masks, edge_masks)]=0
    new_masks=np.unique(new_masks, return_inverse=True)[1].reshape(masks.shape) # renumber masks to consecutive integers with edge masks removed
    return new_masks

def segment_img(img, cp_model, size_model=None, diameter=30, color_channels=[0,0], mend=True, max_gap_size=300, tiled_edge=False, membrane_channel=-1, **kwargs):
    # COMPUTE SEGMENTATION
    if size_model is not None:
        diameter, style_diams=size_model.eval(img, channels=color_channels)
    masks, flows = cp_model.eval(img, diameter=diameter, channels=color_channels, **kwargs)[:2] # segment data, toss outputted styles and diams
    masks=utils.remove_edge_masks(masks)
    if tiled_edge:
        if membrane_channel and img.ndim>2:
            membrane=img[...,membrane_channel]
        else:
            membrane=img
        masks=remove_edge_masks(membrane, masks)
    
    # mend gaps
    if mend:
        masks=preprocessing.mend_gaps(masks, max_gap_size=max_gap_size)[0]
    masks=masks.astype(np.min_scalar_type(masks.max()))
    
    # pull boundary values from masks
    outlines=utils.masks_to_outlines(masks)
    
    outlines_list=utils.outlines_list(masks)
    export={'img':img, 'masks':masks, 'outlines':outlines, 'outlines_list':outlines_list} # my reduced export: just the image, masks, and outlines. Flows, diams, colors etc. are just for cellpose's own reference so I toss them.
    return export

def combine_FUCCI_channels(imgs):
    membrane=imgs[...,2]
    red_bounds=np.quantile(imgs[...,0], [0.01, 0.99])
    green_bounds=np.quantile(imgs[...,1], [0.01, 0.99])

    red=(imgs[...,0]-red_bounds[0])/(red_bounds[1]-red_bounds[0])
    green=(imgs[...,1]-green_bounds[0])/(green_bounds[1]-green_bounds[0])

    nuclei=red+green/2

    imgs=np.stack([nuclei, membrane], axis=-1)
    return imgs
    
def segment_stack(stack_path, output_path=None, segmentation_channel='membrane', geminin_path=None, pip_path=None, initial_frame_number=0, **kwargs):
    # load model
    if segmentation_channel=="membrane" or segmentation_channel=='FUCCI':
        cp_model=models.CellposeModel(gpu=True, model_type='cyto3')
        size_model=models.SizeModel(cp_model, pretrained_size='C:\\Users\\Andy\\.cellpose\\models\\size_cyto3.npy')
    ## TODO: update these
    elif segmentation_channel=="nuclei":
        cp_model=models.CellposeModel(gpu=True, model_type='nuclei')
        size_model=models.SizeModel(cp_model, pretrained_size='C:\\Users\\Andy\\.cellpose\\models\\size_nucleitorch_0.npy')
    if segmentation_channel=="ZO-1":
        cp_model=models.Cellpose(gpu=True, model_type='gastruloid_v6')
        size_model=models.SizeModel(cp_model, pretrained_size='C:\\Users\\Andy\\.cellpose\\models\\size_cyto3.npy')
    
    stack_path=Path(stack_path)
    stack=io.imread(stack_path)

    if segmentation_channel=='FUCCI': # for FUCCI, also load and combine nuclear channels
        pip=io.imread(pip_path)
        geminin=io.imread(geminin_path)
        nuclei=preprocessing.normalize(pip, quantile=(0.01,0.99))+preprocessing.normalize(geminin, quantile=(0.01,0.99))
        membrane=stack
        stack=np.stack([nuclei, stack], axis=-1)

    if not output_path: # standard output folder formatting
        file_stem=stack_path.stem # take name of stack as base name for segmented files
        dataset=stack_path.parts[-2] # stack folder is name of dataset
        output_path='F:/my_data/Segmentation/{}/segmented/{}'.format(dataset, file_stem)

    output_path=Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for frame_number, img in enumerate(tqdm(stack)):
        # COMPUTE SEGMENTATION
        export=segment_img(img, cp_model, size_model, **kwargs)

        if segmentation_channel=='FUCCI': # rewrite the image channel with nuclear channels separated
            export['img']=np.stack([geminin[frame_number], pip[frame_number], membrane[frame_number]], axis=-1)

        np.save(f'{output_path}/{output_path.stem}-{frame_number+initial_frame_number:03}_seg.npy', export)


from cellpose import models
def simple_segment(img, model_type, diameter):
    model=models.CellposeModel(gpu=True, model_type=model_type)
    mask=model.eval(img, diameter=diameter, channels=[0,0])[0]
    mask=utils.remove_edge_masks(mask)
    return mask

from multiprocessing import Pool
def parallel_segment(imgs, model_type, diameter, progress_bar=None):
    from functools import partial
    #model=models.CellposeModel(gpu=True, model_type=model_type)
    p=Pool(processes=8)
    if progress_bar is None:
        progress_bar=lambda x: x
        progress_kwargs={}
    else:
        progress_kwargs={'total':len(imgs), 'desc':'segmenting'}

    out=[x for x in progress_bar(p.imap(partial(simple_segment, model_type=model_type, diameter=diameter), imgs, chunksize=8), **progress_kwargs)]
    return out