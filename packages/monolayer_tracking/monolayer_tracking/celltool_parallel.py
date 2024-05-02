from celltool import simple_interface as ct
from multiprocessing import Pool
from tqdm.notebook import tqdm
import numpy as np
from functools import partial
import pickle
from celltool.contour import contour_class
from natsort import natsorted
from glob import glob
from pathlib import Path

def sample_reference_contour(smoothed_contours, sample_size=2000):
    aligned_contours=align_contours_parallel(np.random.choice(smoothed_contours, size=sample_size), chunk_kwargs={'size':100}, concat=False)

    mean_contours=[]
    for contours in tqdm(aligned_contours):
        pca_contour=ct.make_shape_model(contours, required_variance_explained=0.95)[0]
        mean_contours.append(pca_contour)
    mean_contours=ct.align_contours(mean_contours)
    reference_contour=ct.make_shape_model(mean_contours, required_variance_explained=0.95)[0]
    
    return reference_contour

def export_smoothed_contours(input_path, output_path=None, chunk_kwargs={}, scale=0.3225, units='\N{MICRO SIGN}m', n_resample=100, smoothing=0.007, verbose=False):
    if verbose:
        def vprint(obj):
            print(obj)
    else:
        def vprint(obj):
            pass
        
    if not output_path: # generate a corresponding output path if not specified
        output_path=input_path.replace('segmented','contours')
    print(f'opening {input_path}')
    contours=[]
    vprint('loading contours')
    segmented_files=natsorted(glob(f'{input_path}/*seg.npy'))
    for segmented_file in segmented_files:
        outlines_list = np.load(segmented_file, allow_pickle=True).item()['outlines_list']
        for cell_number, outline in enumerate(outlines_list):
            if len(outline)==0:
                continue
            contour=contour_class.Contour(points=outline)
            contour._filename=f'{output_path}{Path(segmented_file).stem[:-4]}-{cell_number}.contour' # format: '$stage_number-membrane-$timepoint-$cell_number.contour'
            contours.append(contour)
    
    vprint(f'created {len(contours)} contours')
    
    vprint('rescaling contours')
    scaled_contours=transform_contours_parallel(contours, scale_factor=scale, units=units, chunk_kwargs=chunk_kwargs)

    # resample contours
    vprint('resampling contours')
    smoothed_contours=resample_contours_parallel(scaled_contours, resample_points=n_resample, smoothing=smoothing, chunk_kwargs=chunk_kwargs) # this one is the computational bottleneck: 541 contours, ~2 minutes
    
    # export contours
    vprint('exporting contours')
    Path(output_path).mkdir(parents=True, exist_ok=True) # create export folder
    contours_dir=f'{output_path}/smoothed_contours.pkl'
    with open(contours_dir, 'wb') as file:
        pickle.dump(smoothed_contours, file) 
    print(f'completed {output_path}')

def export_aligned_contours(input_path, reference_contour=None, allow_reflection=True, chunk_kwargs={}):
    smoothed_contour_dir=input_path+'/smoothed_contours.pkl'
    with open(smoothed_contour_dir, 'rb') as file: 
        smoothed_contours = pickle.load(file)
    print(f'loaded {len(smoothed_contours)} contours')
    
    print('aligning contours')
    aligned_contours=align_contours_to_parallel(smoothed_contours, reference=reference_contour, chunk_kwargs=chunk_kwargs, allow_reflection=allow_reflection)
    
    print('exporting contours')
    aligned_dir=input_path+'/aligned_contours.pkl'
    with open(aligned_dir, 'wb') as file: 
        pickle.dump(aligned_contours, file)

def chunk_list(list_, n=8, size=None):
    ''' split a list or array of values into a certain number of chunks (if n, no size), or chunks of a set size (if size).'''
    from math import ceil
    if not size:
        size=ceil(len(list_)/n)

    chunked_list=[list_[i:i+size] for i in range(0,len(list_),size)]
    
    return chunked_list, len(chunked_list)

def _contour_parallelize(f, contours, chunk_kwargs={}, n_processors=8, show_tqdm=True, concat=True, **kwargs):
    ''' 
    wrapper to parallelize a celltool function that operates on a list of contours. 
    PARAMS:
    - f (func): function to parallelize
    - contours (array-like): 1D collection of contours
    - chunk_kwargs (dict): either int n or int size to pass to chunk_list()
    '''
    p=Pool(processes = n_processors)

    chunked_contours, n_chunks = chunk_list(contours, **chunk_kwargs)

    if show_tqdm:
        progress_bar=tqdm
        progress_total={'total':n_chunks}
    else:
        progress_bar=lambda x: x
        progress_total={}

    output_contours=[x for x in progress_bar(p.imap(partial(f, **kwargs), chunked_contours), **progress_total)]

    if concat:
        output_contours=np.concatenate(output_contours)

    return output_contours

# can I do this with decorators?
def transform_contours_parallel(*args, **kwargs):
    return _contour_parallelize(ct.transform_contours, *args, **kwargs)

def resample_contours_parallel(*args, **kwargs):
    return _contour_parallelize(ct.resample_contours, *args, **kwargs)

def align_contours_parallel(*args, **kwargs):
    return _contour_parallelize(ct.align_contours, *args, **kwargs)

def align_contours_to_parallel(*args, **kwargs):
    return _contour_parallelize(ct.align_contours_to, *args, **kwargs)
