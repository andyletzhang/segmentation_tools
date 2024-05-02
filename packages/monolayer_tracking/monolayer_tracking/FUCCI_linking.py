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