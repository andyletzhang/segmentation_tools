import numpy as np
import pandas as pd


def impute_fill(cell_cycle, limit=6):
    cell_cycle=cell_cycle.replace(0,np.nan).ffill(limit=limit)
    cell_cycle=cell_cycle.bfill(limit=limit).fillna(0).astype(int)
    return cell_cycle

def smooth_small_flickers(cell_cycle):
    oneHot_cc=pd.get_dummies(cell_cycle)
    smoothed_cc=np.nansum([oneHot_cc.shift(-1), oneHot_cc, oneHot_cc.shift(1)], axis=0)/3
    smoothed_cc[[0,-1]]*=3/2
    cell_cycle=oneHot_cc.columns[np.argmax(smoothed_cc, axis=1)]
    return pd.Series(cell_cycle)

def get_transitions(cell_cycle):
    is_transition=[True,*(np.diff(cell_cycle)!=0)]
    stages=cell_cycle.reset_index(drop=True)[is_transition]
    stage_lengths=cell_cycle.groupby(np.cumsum(is_transition)).size()
    return stages, stage_lengths

def correct_mother_G2(cell_cycle):
    stages, stage_lengths=get_transitions(cell_cycle)
    if np.array_equal(stages[-2:], [3,2]):
        last_stage_length=stage_lengths.iloc[-1]
        if last_stage_length<=6: # if it's red for longer than an hour, probably not a mitosis thing
            cell_cycle[-last_stage_length:]=3 # correct to G2
    return cell_cycle

def correct_daughter_G1(cell_cycle):
    stages, stage_lengths=get_transitions(cell_cycle.replace(2,3)) # get transitions, with S and G2 grouped into one stage
    if np.array_equal(stages[:2], [3,1]): # check for transitions from S/G2 -> G1 immediately after cell shows up
        first_stage_length=stage_lengths.iloc[0]
        if first_stage_length>=4: # if it's red for longer than an hour, probably not a mitosis thing
            cell_cycle[:first_stage_length]=1 # correct to G2
    return cell_cycle

def subsequence(seq, subseq):
    '''Returns list of indices where the subsequence is found in the sequence (indexed first element). If not found, returns an empty list.'''
    seq_length=len(seq)
    subseq_length=len(subseq)
    return np.array([x for x in range(seq_length-subseq_length+1) if np.array_equal(seq[x:x+len(subseq)], subseq)])
    
def remove_sandwich_flicker(cell_cycle, transition):
    corrected=False
    stages, stage_lengths=get_transitions(cell_cycle)
    flickers=subsequence(stages, transition)

    for flicker in flickers:
        outer_length=stage_lengths.iloc[[flicker,flicker+2]].sum()
        inner_length=stage_lengths.iloc[flicker+1]
        if outer_length>inner_length:
            false_start, false_end=stages.index[flicker+1:flicker+3]
            cell_cycle[false_start:false_end]=transition[-1] # correct to outer value
            corrected=True

    if corrected: # made a correction, should check if there's something new that can be fixed
        cell_cycle=remove_sandwich_flicker(cell_cycle, transition)

    return cell_cycle

# find FUCCI info that doesn't make sense over time
def get_problematic_IDs(trajectories):
    problematic_IDs=[]
    for particle_ID, cc_stage in trajectories.groupby('particle')['cell_cycle']:
        cc_transitions=np.diff(cc_stage)
        if np.any(~((cc_transitions==0)|(cc_transitions==1))): # a transition that's not G1->S or S->G2
            problematic_IDs.append(particle_ID)

    return(problematic_IDs)

def bg_subtract_nuclei(red, green, nuclear_masks,blur_sigma=2, bin_size=100, interp_factor=8):
    import background_tps as tps
    from scipy import ndimage

    masked_red=red.copy().astype(float)
    masked_red[nuclear_masks>0]=np.nan
    red_bg=tps.tps_bg(masked_red, bin_size=bin_size, interp_factor=interp_factor)
    red_subtracted=red-red_bg+np.std(red)
    if blur_sigma>0:
        red_subtracted_blurred=ndimage.gaussian_filter(red_subtracted, sigma=blur_sigma)
    else:
        red_subtracted_blurred=red_subtracted
    red_subtracted_blurred[red_subtracted_blurred<0]=0

    masked_green=green.copy().astype(float)
    masked_green[nuclear_masks>0]=np.nan
    green_bg=tps.tps_bg(masked_green, bin_size=bin_size, interp_factor=interp_factor)
    green_subtracted=green-green_bg+np.std(green)
    if blur_sigma>0:
        green_subtracted_blurred=ndimage.gaussian_filter(green_subtracted, sigma=blur_sigma)
    else:
        green_subtracted_blurred=green_subtracted
    green_subtracted_blurred[green_subtracted_blurred<0]=0

    return red_subtracted_blurred, green_subtracted_blurred

def get_fucci_percentages(red, green, nuclear_masks, red_threshold=1, green_threshold=1):
    from scipy import ndimage

    nuc_green = ndimage.median(green, labels=nuclear_masks, index=np.arange(1, nuclear_masks.max()+1))
    nuc_red = ndimage.median(red, labels=nuclear_masks, index=np.arange(1, nuclear_masks.max()+1))

    red_percentage=nuc_red/np.percentile(red[nuclear_masks==0], 95)
    green_percentage=nuc_green/np.percentile(green[nuclear_masks==0], 95)

    return red_percentage, green_percentage

def points_to_masks(masks, centroids):
    centroid_integers=centroids.round().astype(int)

    pairs=masks[centroid_integers[:,0], centroid_integers[:,1]]
    matches=np.stack((np.arange(len(pairs)), pairs-1), axis=1)
    unpaired_centroids=np.where(pairs==0)[0]
    matches=matches[pairs!=0] # drop nuclei without membrane
    matches=matches[~pd.Series(matches[:,1]).duplicated(keep='first')]

    unpaired_masks = [n for n in range(1, masks.max()+1) if n not in matches[:,1]+1]

    return matches, unpaired_centroids, unpaired_masks