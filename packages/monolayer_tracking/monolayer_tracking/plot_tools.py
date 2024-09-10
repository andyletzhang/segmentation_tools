import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.notebook import tqdm

from monolayer_tracking import preprocessing

def plot_trajectories(stack, output_path=None, trail_length=10, linewidth=1, markersize=1, figsize=10, show_labels=True, tracking_kwargs={}):
    '''display trajectories superimposed over images.'''
    from matplotlib import collections
    from matplotlib.patches import Polygon
    
    if not hasattr(stack, 'tracked_centroids'):
        stack.track_centroids(**tracking_kwargs)
    trajectories=stack.tracked_centroids
    
    if not output_path:
        output_path=stack.name.replace('segmented','tracking')
    output_path=Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for n, frame in enumerate(tqdm(stack.frames)):
        frame_drift=stack.drift.loc[n]
        aspect=np.divide(*frame.img.shape[:2])
        plt.figure(figsize=(figsize, figsize*aspect))
        ax=plt.axes()
        
        if len(frame.img.shape)==2:
            plt.imshow(preprocessing.normalize(frame.img), cmap='gray')
        elif len(frame.img.shape)==3:
            if frame.img[...,2].max()>0:
                plt.imshow(preprocessing.normalize(frame.img[...,2]), cmap='gray') # FUCCI data, blue is membrane channel
            else:
                plt.imshow(preprocessing.normalize(frame.img)) # red nuclei, green membrane
        else:
            raise ValueError(f'frame.img has an unexpected number of dimensions {frame.img.ndim}. Can take 2 (grayscale) or 3 (color) dimensions.')
        
        trails=[] # particle trajectories as plotted on each frame
        for particle, trajectory in trajectories[trajectories.frame<=n].groupby('particle'): # get tracking data up to the current frame for one particle at a time
            trails.append(Polygon(trajectory[['x','y']].iloc[-trail_length:]+frame_drift,
                                  closed=False, linewidth=linewidth*figsize/5, fc='None', ec='C{}'.format(particle)))

            if show_labels:
                in_frame=trajectory.index[trajectory['frame']==n] # get the cell number of the particle in this frame
                try: # try labeling this particle in this frame
                    cell_number=in_frame[0] # cell number
                    # TODO: I think the bottleneck is indexing and plotting every text label by hand: can I speed this up somehow? (or I just don't do this too often)
                    particle_label=plt.text(trajectory['x'].loc[cell_number]+5+frame_drift['x'],trajectory['y'].loc[cell_number]-5+frame_drift['x'],particle, color='white', fontsize=figsize/2)
                    particle_label.set_path_effects([patheffects.Stroke(linewidth=figsize/5, foreground='C{}'.format(particle)),patheffects.Normal()])

                except IndexError: # indexerror means the particle's not identifiable in this frame, move on
                    continue

        # draw trails
        ax.add_artist(collections.PatchCollection(trails, match_original=True))
        
        # add centroids
        plt.scatter(frame.centroids()[:,1],frame.centroids()[:,0],s=np.sqrt(figsize/5)*markersize,c='white')
        
        # cosmetics
        xlim=np.array([0,frame.resolution[1]])+frame_drift['x']
        ylim=np.array([frame.resolution[0], 0])+frame_drift['y']
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.axis('off')

        plt.savefig(output_path/f'{output_path.stem}-{n}.tif',dpi=300)
        plt.close()


def highlight_mitoses(stack, output_path=None, show_labels=True, figsize=6):
    """
    Highlight mitotic events in each frame of a stack, and output tifs.

    Args:
        stack: A Stack object containing information about frames and mitotic events.
        output_path (str or Path): The directory where the highlighted images will be saved. If not provided, the default directory will be used.

    Returns:
        None
    """
    # Ensure mitoses are calculated and tracked centroids are available
    if not hasattr(stack, 'mitoses'):
        stack.get_mitoses()
    if not hasattr(stack, 'tracked_centroids'):
        stack.track_centroids()
    
    # Initialize list to store mitosis information over time
    extended_mitoses = []

    # Iterate over each mitotic event in the stack
    for m in stack.mitoses:
        # Extract particle IDs for the mother and daughters
        particle_IDs = m.index.get_level_values(0)
        
        # Extract data for the mother cell
        mother = stack.tracked_centroids[stack.tracked_centroids['particle'] == particle_IDs[0]]
        mother_firstframe = mother.frame.max() - 5
        mother = mother[mother['frame'] >= mother_firstframe]
        mother['alpha'] = (mother.frame - mother.frame.min() + 1) / 6  # Calculate alpha for plotting
        mother['color'] = 'r'  # Set color for plotting
        extended_mitoses.append(mother)  # Append to extended mitoses list
        
        # Extract data for each daughter cell
        for particle_ID in particle_IDs[1:]:
            daughter = stack.tracked_centroids[stack.tracked_centroids['particle'] == particle_ID]
            daughter_firstframe = daughter.frame.min() + 5
            daughter = daughter[daughter['frame'] <= daughter_firstframe]
            daughter['alpha'] = (daughter.frame.max() - daughter.frame + 1) / 6  # Calculate alpha for plotting
            daughter['color'] = 'lime'  # Set color for plotting
            extended_mitoses.append(daughter)  # Append to extended mitoses list

    extended_mitoses = pd.concat(extended_mitoses)
    extended_mitoses = extended_mitoses.set_index(['particle','frame'])
    
    # Extract graphical cell centroids (not drift-corrected)
    cell_centroids = []
    for cell in extended_mitoses.reset_index()[['frame','cell_number']].iterrows():
        cell_centroids.append(stack.frames[cell[1]['frame']].centroids()[cell[1]['cell_number']])
    extended_mitoses[['y','x']] = pd.DataFrame(cell_centroids, columns=['y','x'], index=extended_mitoses.index)

    # Iterate over each frame in the stack
    for frame_number, frame in enumerate(tqdm(stack.frames)):
        if not hasattr(frame,'img'):
            frame.load_img()

        if frame.img.ndim==3: # different RGB conventions, just by how I typically save img data
            if frame.img[...,2].max()>0:
                img=frame.img[...,2] # FUCCI data, blue is membrane channel
            else:
                img=frame.img # red nuclei, green membrane
        else:
            img=frame.img
        aspect = np.divide(*img.shape[:2])
        plt.figure(figsize=(figsize, figsize*aspect))
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        # Label the two mitosis-identified frames with cell outlines and centroids
        if np.any(extended_mitoses.index.get_level_values(1) == frame_number):
            cells_to_outline = extended_mitoses.xs(frame_number, level='frame')
            plt.scatter(cells_to_outline['x'], cells_to_outline['y'], s=2, color='white')
            for index, cell in cells_to_outline.iterrows():
                cell_number = int(cell['cell_number'])
                if show_labels:
                    # Add cell label
                    cell_label = plt.text(frame.cells[cell_number].centroid[1], frame.cells[cell_number].centroid[0], cell_number, color=cell['color'], fontsize=5)
                    cell_label.set_path_effects([patheffects.Stroke(linewidth=1, foreground='k'), patheffects.Normal()])
                # Plot cell outline
                plt.plot(*frame.cells[cell_number].outline.T, alpha=cell['alpha'], color=cell['color'], linewidth=0.5)
            
        # Set the output path for saving images
        if not output_path:
            output_path = stack.name.replace('segmented', 'mitosis_highlight')
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save the plot as an image
        plt.savefig(output_path / f'{output_path.stem}-{frame_number}.tif', dpi=300)
        plt.close()


# new, gradient demo seg
def demo_seg(stack, output_path=None, highlight_mitoses=True, figsize=6, polygon_ax=True):
    with plt.style.context('dark_background'):
        if highlight_mitoses:
            if not hasattr(stack, 'mitoses'):
                stack.get_mitoses()

            extended_mitoses, daughter_CoMs, death_frames = [], [], []
            for m in stack.mitoses:
                graphical_centroids=[stack.frames[frame].centroids()[cell_number] for frame, cell_number in np.array(m.iloc[1:].reset_index()[['frame','cell_number']])] # pull centroids of daughters, not drift-corrected
                daughter_CoMs.append(np.flip(np.mean(graphical_centroids, axis=0))) # get daughter CoM (np.flipped to x,y order for plotting)
                death_frames.append(m.index.get_level_values(1)[0])
                particle_IDs=m.index.get_level_values(0)

                mother=stack.tracked_centroids[stack.tracked_centroids['particle']==particle_IDs[0]]
                mother_firstframe=mother.frame.max()-5
                mother=mother[mother['frame']>=mother_firstframe]
                mother['alpha']=(mother.frame-mother.frame.min()+1)/6
                mother['color']='r'
                extended_mitoses.append(mother)

                for daughter_ID in particle_IDs[1:]:
                    daughter=stack.tracked_centroids[stack.tracked_centroids['particle']==daughter_ID]
                    daughter_firstframe=daughter.frame.min()+5
                    daughter=daughter[daughter['frame']<=daughter_firstframe]
                    daughter['alpha']=(daughter.frame.max()-daughter.frame+1)/6
                    daughter['color']='lime'
                    extended_mitoses.append(daughter)

            extended_mitoses=pd.concat(extended_mitoses)
            extended_mitoses=extended_mitoses.set_index(['particle','frame'])
            daughter_CoMs=pd.DataFrame(daughter_CoMs, columns=['x','y'], index=death_frames)

            # fetch graphical cell centroids (not drift-corrected)
            cell_centroids=[]
            for cell in extended_mitoses.reset_index()[['frame','cell_number']].iterrows():
                cell_centroids.append(stack.frames[cell[1]['frame']].centroids()[cell[1]['cell_number']])
            extended_mitoses[['y','x']]=pd.DataFrame(cell_centroids, columns=['y','x'], index=extended_mitoses.index)
        
        for n, frame in enumerate(tqdm(stack.frames)):
            frame.get_TCJs()
            if not hasattr(frame,'img'):
                frame.load_img()

            if frame.img.ndim==3: # different RGB conventions, just by how I typically save img data
                if frame.img[...,2].max()>0:
                    img=frame.img[...,2] # FUCCI data, blue is membrane channel
                else:
                    img=frame.img # red nuclei, green membrane
            else:
                img=frame.img
            aspect = np.divide(*img.shape[:2])

            if polygon_ax:
                n_axes=3
            else:
                n_axes=2
            fig, axes = plt.subplots(1, n_axes, sharey=True, sharex=True, figsize=(figsize*n_axes, figsize*aspect))

            if img.ndim==2:
                axes[0].imshow(preprocessing.normalize(img), cmap='gray')
            else:
                axes[0].imshow(preprocessing.normalize(img))
                
            axes[0].set_title('image data')
            axes[0].set_xticks([])
            axes[0].set_yticks([])

            axes[1].imshow(frame.outlines.todense()!=0,cmap='gray')
            axes[1].set_title('segmented cell outlines')
            axes[1].set_xticks([])
            axes[1].set_yticks([])

            if polygon_ax: # put under line 258 using zorder
                axes[2].add_collection(frame.cell_polygons(fc='pink',linewidths=0.5))
            
            if highlight_mitoses:
                # redraw mitotic cells in red and green
                cell_polygons, facecolors, alphas = [], [], []
                if np.any(extended_mitoses.index.get_level_values(1)==n):
                    cells_to_outline=extended_mitoses.xs(n,level='frame')
                    for _, cell in cells_to_outline.iterrows():
                        axes[0].scatter(cell['x'], cell['y'], alpha=cell['alpha'], fc=cell['color'], ec='k', s=figsize/4, linewidth=0.2)
                        axes[1].scatter(cell['x'], cell['y'], alpha=cell['alpha'], fc=cell['color'], ec='k', s=figsize/4, linewidth=0.2)
                        try:
                            vertices=frame.cells[cell['cell_number']].sorted_vertices
                        except AttributeError:
                            continue
                        vertices=np.array([vertices[:,1],vertices[:,0]]).T # flip x and y for plotting
                        if polygon_ax:
                            cell_polygons.append(Polygon(vertices))
                        facecolors.append(cell['color'])
                        alphas.append(cell['alpha'])
                    if polygon_ax:
                        if len(cell_polygons)>0:
                            mitotic_polygons=PatchCollection(cell_polygons, edgecolors='k',facecolors=facecolors, alpha=alphas, linewidths=0.5)
                            axes[2].add_collection(mitotic_polygons)
            if polygon_ax:
                axes[2].set_title('vertex reconstruction')
                axes[2].set_xticks([])
                axes[2].set_yticks([])
                axes[2].set_aspect('equal')
            
            if not output_path:
                output_path = stack.name.replace('segmented', 'demo_seg')
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            # Save the plot as an image
            plt.savefig(output_path / f'{output_path.stem}-{n}.tif', dpi=300)
            plt.close()

def FUCCI_overlay(frame, imshow=True, ax=None, show_denoised=True, show_labels=False, alpha=0.2, normalize=True):
    if not ax: ax=plt.gca()
    if imshow:
        if show_denoised:
            color_FUCCI=np.stack([frame.FUCCI[0], frame.FUCCI[1], np.zeros(frame.masks.shape, dtype=int)], axis=-1)
        else:
            if not hasattr(frame, 'img'):
                frame.load_img()
            if normalize:
                norm=preprocessing.normalize
            elif norm==False:
                norm=lambda x: x
            else:
                norm=normalize
            color_FUCCI=np.stack([norm(frame.img[...,0]), norm(frame.img[...,1]), np.zeros(frame.masks.shape, dtype=int)], axis=-1)
        plt.imshow(color_FUCCI)
    else:
        plt.xlim(0,frame.masks.shape[1])
        plt.ylim(frame.masks.shape[0],0)

    if show_labels:
        frame.get_centroids()

    cell_overlays=[]
    colors={0:'none', 1:'lime', 2:'r', 3:'orange'}
    for cell in frame.cells:
        
        cell_overlays.append(Polygon(cell.outline, ec='white', fc=colors[cell.cycle_stage], linewidth=1, alpha=alpha))

        if show_labels:
            particle_label=plt.text(cell.centroid[1],cell.centroid[0],s=cell.n, color='white', fontsize=6)
            particle_label.set_path_effects([patheffects.Stroke(linewidth=1, foreground=colors[cell.cycle_stage]),patheffects.Normal()])

    ax.add_artist(PatchCollection(cell_overlays, match_original=True))
    plt.axis('off')

# Volume Plots
def volume_boxplot(volumes, labels=None, ax=None, SC_color='C0', ME_color='C2', default_color='C1', **boxplot_kwargs):
    '''
    Simple boxplot of volumes. Cell cycle dimension is concatenated. SC and WT are colored blue and green, respectively.

    Parameters
    ----------
    volumes (dict or list): Each entry is a list of volumes for respective cell cycle sizes (G1, S, G2).
    labels (list of strings): Labels for each group.
    ax (matplotlib axis): If None, current axis is used.
    SC_color (str): Color for SC data.
    ME_color (str): Color for ME data.
    default_color (str): Color for other data.
    boxplot_kwargs (dict): Additional arguments for plt.boxplot.
    '''

    if ax==None: ax=plt.gca()
    if labels is None: labels=['' for _ in volumes]
    
    default_kwargs={'showfliers':False, 'patch_artist':True, 'notch':True, 'vert':True}
    boxplot_kwargs=default_kwargs|boxplot_kwargs # Merge default and user kwargs

    if isinstance(volumes, dict):
        volumes=[np.concatenate(volumes[label]) for label in labels]
    else:
        volumes=[np.concatenate(vol) for vol in volumes]

    bp=ax.boxplot(volumes, labels=labels, **boxplot_kwargs)

    for patch, label in zip(bp['boxes'], labels):
        if label.startswith('SC'):
            color=SC_color
        elif label.startswith('ME'):
            color=ME_color
        else:
            color=default_color
        patch.set_facecolor(color)

    ax.set_xlabel('Volume (μm$^3$)')

def cell_cycle_boxplot(volumes, labels=None, ax=None, colors=None, hide_NS=True, ctrl_line=False, xticks='n', trial_labels=True, **boxplot_kwargs):
    '''
    Boxplot of volumes for each cell cycle phase.

    Parameters
    ----------
    volumes (dict or list): Each entry is a list of volumes for respective cell cycle sizes (G1, S, G2).
    labels (list of strings): Labels for each group.
    ax (matplotlib axis): If None, current axis is used.
    colors (list of colors): Colors for each cell cycle phase.
    ctrl_line (str or bool): Draws a dashed line at the median values of the control dataset. If True, WT is used as control. If str, all labels including the string are averaged.
    xticks (str): If 'n', xticks are the number of cells in each group. If 'cycle', xticks are the cell cycle phases.
    trial_labels (bool or list): labels above each group. If True, labels are the same as labels. If list, labels are the list.
    boxplot_kwargs (dict): Additional arguments for plt.boxplot.
    '''

    if not ax: ax=plt.gca()
    if labels is None: labels=['' for _ in volumes]

    if isinstance(volumes, dict):
        volumes=[volumes[label] for label in labels]

    default_kwargs={'showfliers':False, 'notch':True, 'vert':True}
    boxplot_kwargs=default_kwargs|boxplot_kwargs

    all_bps=[]
    for i, vol, label in zip(range(len(labels)), volumes, labels):
        if hide_NS:
            vol=vol[-3:]
        has_NS=not hide_NS and len(vol)==4
        if colors is None:
            colors=['g','r','orange']
            if has_NS:
                colors=['k']+colors

        if has_NS:
            positions=np.arange(i*5, i*5+4)
        else:
            positions=np.arange(i*4, i*4+3)
        
        if xticks=='n':
            box_labels=[f'n={len(v)}' for v in vol]
        elif xticks=='cycle':
            box_labels=['G1','S','G2']
            if has_NS:
                box_labels=['NS']+box_labels
        else:
            box_labels=['' for _ in range(len(vol))]

        bp=ax.boxplot(vol, positions=positions, patch_artist=True, labels=box_labels, **boxplot_kwargs)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        all_bps.append(bp)
    
    if trial_labels:
        ylim=ax.get_ylim()
        y_increment=0.1*(ylim[1]-ylim[0])
        ax.set_ylim(ylim[0], ylim[1]+y_increment)
        if trial_labels==True:
            trial_labels=labels
        for i, label in enumerate(trial_labels):
            ax.text(i*4+1, ylim[1], label, ha='center', va='center', weight='bold')
    
    if ctrl_line:
        if ctrl_line==True:
            ctrl_line='WT'
        ctrl_vols=[volumes[i] for i in np.where([label.startswith(ctrl_line) for label in labels])[0]]
        if len(ctrl_vols)==0:
            print('No WT labels found in the list')
        else:
            median_values=[np.median(np.concatenate([vol[i] for vol in ctrl_vols])) for i in range(3)]
            for med_value, color in zip(median_values, colors):
                ax.axhline(med_value, color=color, linestyle='--', zorder=0)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 50)
    ax.set_ylabel('Volume (μm$^3$)')

    return all_bps

def cell_cycle_occupancy_barplot(volumes, labels=None, hide_NS=True, ax=None, xticks='n', colors=None, edge_color='k', **barplot_kwargs):
    '''
    Barplot of occupancy for each cell cycle phase.

    Parameters
    ----------
    volumes (dict or list): Each entry is a list of volumes for respective cell cycle sizes (G1, S, G2).
    labels (list of strings): Labels for each group.
    ax (matplotlib axis): If None, current axis is used.
    xticks (str): If 'n', xticks are the number of cells in each group. If 'cycle', xticks are the cell cycle phases.
    colors (list of colors): Colors for each cell cycle phase.
    edge_color (str): Edge color for the bars.
    barplot_kwargs (dict): Additional arguments for plt.bar.
    '''
    if not ax: ax=plt.gca()
    if labels is None: labels=['' for _ in volumes]
    
    if hide_NS:
        volumes=[vol[-3:] for vol in volumes]
    if isinstance(volumes, dict):
        volumes=[volumes[label] for label in labels]

    occupancies=[[len(v) for v in vol] for vol in volumes]
    total_occupancies=[np.sum(o) for o in occupancies]
    percent_occupancies=np.concatenate([np.array(o)/np.sum(o) for o in occupancies])

    has_NS=not hide_NS and len(volumes[0])==4
    condition_spacing=5 if has_NS else 4

    if colors is None:
        colors=['g','r','orange']
        if has_NS:
            colors=['k']+colors

    positions=np.concatenate([np.arange(i*condition_spacing, (i+1)*condition_spacing-1) for i in range(len(labels))])

    bars=ax.bar(positions, percent_occupancies, **barplot_kwargs)

    for i in range(condition_spacing-1):
        for bar in bars[i::condition_spacing-1]: bar.set_color(colors[i])

    for bar in bars: bar.set_edgecolor(edge_color)
    if xticks=='n':
        ax.set_xticks((np.arange(len(labels))+1/2)*(condition_spacing)-1, [f'n={o}' for o in total_occupancies])
    elif xticks=='cycle':
        cc_labels=['G1','S','G2']
        if has_NS:
            cc_labels=['NS']+cc_labels
        ax.set_xticks(positions, cc_labels*len(labels))
    return bars
        
def cell_cycle_plot(volumes, labels=None, axes=None, figsize=(6,6), hide_NS=True, gridspec_kw={'height_ratios':[6,1]}, sharex=True, boxplot_kwargs={}, barplot_kwargs={}, subplot_kwargs={}):
    '''
    Wrapper function for plotting cell cycle data. Plots volume boxplot and occupancy barplot.

    Parameters
    ----------
    volumes (dict or list): Each entry is a list of volumes for respective cell cycle sizes (G1, S, G2).
    labels (list of strings): Labels for each group.
    axes (list of matplotlib axes): If None, new axes are created.
    figsize (tuple): Figure size.
    gridspec_kw (dict): Arguments for gridspec_kw.
    sharex (bool): If True, x-axis is shared.
    boxplot_kwargs (dict): Additional arguments for volume_boxplot.
    barplot_kwargs (dict): Additional arguments for occupancy_barplot.
    subplot_kwargs (dict): Additional arguments for plt.subplots.
    '''
    if not axes:
        fig, axes=plt.subplots(2, 1, figsize=figsize, sharex=sharex, gridspec_kw=gridspec_kw, **subplot_kwargs)
    
    bps=cell_cycle_boxplot(volumes, labels, ax=axes[0], hide_NS=hide_NS, **boxplot_kwargs)
    bars=cell_cycle_occupancy_barplot(volumes, labels, ax=axes[1], hide_NS=hide_NS, **barplot_kwargs)

    return fig, axes, bps, bars

def hist_median(v, histtype='step', weights='default', zorder=None, ax=None, bins=30, range=(0,6000), linewidth=1.4, alpha=1, **kwargs):
    if not ax:
        ax=plt.gca()
    if weights=='default':
        hist_weights=np.ones_like(v)/len(v)
    else:
        hist_weights=weights
    n, bins, patch=ax.hist(v, bins=bins, range=range, zorder=zorder, histtype=histtype, weights=hist_weights, linewidth=linewidth, alpha=alpha, **kwargs)
    median=np.nanmedian(v)
    
    bin_idx=np.digitize(median, bins)-1 # find which bin the median is in
    ax.plot([median, median], [0, n[bin_idx]], color=patch[0].get_edgecolor(), zorder=zorder, linestyle='--', linewidth=linewidth,  alpha=alpha) # draw a line at the median up to the height of the bin