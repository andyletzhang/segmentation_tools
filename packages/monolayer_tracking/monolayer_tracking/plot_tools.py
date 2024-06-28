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
            color_FUCCI=np.stack([frame.img[...,0], frame.img[...,1], np.zeros(frame.masks.shape, dtype=int)], axis=-1)
            if normalize:
                color_FUCCI=preprocessing.normalize(color_FUCCI, quantile=(0.01,0.99))
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