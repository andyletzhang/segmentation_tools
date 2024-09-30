import numpy as np
import pandas as pd
from scipy import ndimage
from natsort import natsorted
from glob import glob
from pathlib import Path
from monolayer_tracking import preprocessing # from my module

'''
    Stacks are collections of time lapse images on a single stage.
    Images are individual seg.npy files, presumed to be segmented membranes generated with cellpose.
    Cells are generated for each cell mask identified by segmentation.
'''

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

class TimeSeries:
    '''
    a stack of images as Image objects, identified as a series of seg.npy files in the same folder and presumed to be in chronological order when sorted by name.
    '''
    def __init__(self, stack_path=None, frame_paths=None, coarse_grain=1, first_frame=0, last_frame=None, verbose_load=False, progress_bar=lambda x: x, **kwargs):
        '''takes the path to the stack and sets up the Stack, an array of Images representing each slice.'''
        if stack_path:
            self.frame_paths=natsorted(glob(stack_path+'/*seg.npy'))[0::coarse_grain] # find and alphanumerically sort all segmented images at the given path
            if len(self.frame_paths)==0:
                raise FileNotFoundError('No seg.npy files found at {}'.format(stack_path)) # warn me if I can't find anything (probably a formatting error)
            if verbose_load:
                print(f'{len(self.frame_paths)} segmented files found at {stack_path}, loading...')
            self.frames=np.array([Image(frame_path, frame_number=n, **kwargs) for n, frame_path in enumerate(progress_bar(self.frame_paths))]) # load all segmented images
            if not stack_path.endswith('/') or not stack_path.endswith('\\'):
                stack_path+='/'
            self.name=stack_path

        elif frame_paths:
            self.frames=np.array([Image(frame_path, frame_number=n, **kwargs) for n, frame_path in enumerate(progress_bar(frame_paths))]) # load all segmented images
            self.name=str(Path(frame_paths[0]).parent)+'/'
            
        else:
            raise ValueError('Either stack_path or frame_paths must be provided.')
    
     # -------------Particle Tracking-------------
    def track_centroids(self, memory=3, v_quantile=0.97, filter_stubs=False, **kwargs):
        '''
        uses the trackpy package to track cell centroids.
        This works in two stages: 
            1. a first tracking with a large search range which identifies and corrects drift and determines a characteristic velocity, and
            2. a second, more discerning tracking with a search range based on the maximum cell speed.
        **kwargs go to tp.link(): 'search_range' depends on cell diameter, and 'memory' can account for the disappearance of a centroid for a certain number of frames (default 0). 
        '''
        import trackpy as tp
        tp.quiet()
        # preliminary search radius for the first pass at tracking. This doesn't have to be perfect, just in the right ballpark (too high better than too low).
        # shifts in the FOV which I've observed are up to ~16 pixels large. Any tracking that wants to accomodate for these must be >16.
        max_search_range=int(np.nanmean([np.sqrt(np.quantile(frame.cell_areas(scaled=False), 0.9)) for frame in self.frames])*2/3) # 2/3 a large cell: overshoot the largest tracking range by a bit.
        
        # FIRST PASS: large search range
        t_firstpass=tp.link(self.centroids(), search_range=max_search_range, adaptive_step=0.96, adaptive_stop=2) # use adaptive search range to find the most tracks without throwing an error

        drift_firstpass=tp.compute_drift(t_firstpass)
        t_corrected=tp.subtract_drift(t_firstpass,drift_firstpass).reset_index(drop=True) # subtract drift in the stage. NB: will also catch any collective migration.
    
        if 'search_range' in kwargs and kwargs['search_range'] is not None: # user can specify a search range if they want. Otherwise one is deduced from the velocity distribution.
            self.tracking_range=kwargs['search_range']
            kwargs.pop('search_range')

        else: # get a search range by evaluating the velocity distribution.
            # This is a simplified copy of the get_velocities() method.
            v=t_corrected[['x','y','frame','particle']].groupby('particle').apply(np.diff, axis=0)
            v_arr=np.concatenate(v)[:,:3]
            velocities=np.linalg.norm(v_arr[:,:2], axis=1)/v_arr[:,2] # get magnitude of velocity normalized by time

            self.tracking_range=np.quantile(velocities, v_quantile) # let's assume the top (1-v_quantile) percent of velocities are spurious tracks or actually mitosis. We'll use this as a cutoff for the real tracking.
        
        # FINAL PASS: small search range
        t_final=tp.link(t_corrected, search_range=self.tracking_range, memory=memory, **kwargs)
        if filter_stubs:
            t_final=tp.filter_stubs(t_final, filter_stubs) # drop tracks which are $filter_stubs or fewer frames long
            t_final['particle']=t_final.groupby('particle').ngroup() # renumber particle tracks with filtered tracks ommitted from count
        drift_finalpass=tp.compute_drift(t_final)
        self.tracked_centroids=tp.subtract_drift(t_final,drift_finalpass).reset_index(drop=True)
        self.drift=drift_firstpass+drift_finalpass
        self.drift=pd.concat([pd.DataFrame({'frame':[0],'y':[0],'x':[0]}).set_index('frame'), self.drift]) # add a row for the first frame (no drift)
        return self.tracked_centroids

    def merge_particle_tracks(self, first_ID, second_ID, frame):
        """
        Merge two particle tracks into a single track.

        Args:
            first_ID (int): ID of the first particle to merge.
            second_ID (int): ID of the second particle to merge.
            frame (int): Frame number at which second_ID should be merged into first_ID.

        Returns:
            DataFrame: DataFrame containing the merged particle track.
        """
        t=self.tracked_centroids
        first_particle=t.loc[t.particle==first_ID]
        second_particle=t.loc[t.particle==second_ID]
        # if necessary, split particles at the frame of interest
        if np.any([first_particle.frame>=frame]):
            self.split_particle_track(first_ID, frame)
        if np.any([second_particle.frame<frame]):
            second_ID=self.split_particle_track(second_ID, frame) # reassign second_ID in case of split

        # merge particles
        t.loc[t.particle==second_ID, 'particle']=first_ID

        # renumber particles so that they're contiguous
        particles=np.unique(t['particle'])
        t['particle']=np.searchsorted(particles, t['particle'])

        return t

    def split_particle_track(self, particle_ID, split_frame):
        """
        Split a particle track into two separate tracks at a given frame.

        Args:
            particle_ID (int): ID of the particle to split.
            split_frame (int): Frame number at which to split the particle track.
            **kwargs: Additional keyword arguments passed to trackpy.link() function.
        
        Returns:
            int: ID of the new particle track created by the split.
        """
        t=self.tracked_centroids
        new_particle_ID=t['particle'].max()+1
        t.loc[(t.particle==particle_ID)&(t.frame>=split_frame), 'particle']=new_particle_ID

        return new_particle_ID

    def get_particle(self, input_ID):
        """
        Retrieve the cell trajectories for a given particle ID or cell.

        Args:
            input_ID (int, np.integer, Cell): ID of the particle or a Cell object.

        Returns:
            list: List of Cell objects in time order representing the trajectory of the particle.
        """
        t = self.tracked_centroids
        
        if isinstance(input_ID, (int, np.integer)):  # If the input is already a particle ID, proceed
            particle_ID = input_ID
        elif isinstance(input_ID, Cell):  # If input is a Cell object, find its corresponding particle ID
            particle_ID = t[(t['frame'] == input_ID.frame) & (t['cell_number'] == input_ID.n)]['particle'].iloc[0]
        elif input_ID is None:  # If no input is provided, return an empty list
            return []
        else:
            raise ValueError(f'Invalid input for get_particle function {input_ID}. Please provide a valid particle ID or Cell object.')

        cell_rows = t[t.particle == particle_ID][['frame', 'cell_number']] # Locate particle

        # Build list of particle appearances over time
        particle = []
        for frame_number, cell_number in np.array(cell_rows):
            particle.append(self.frames[frame_number].cells[cell_number])

        return particle
    
    def get_particle_attr(self, input_ID, attributes, fill_value=None):
        """
        Retrieve a specific attribute for a given particle ID or cell.

        Args:
            input_ID (int, np.integer, Cell): ID of the particle or a Cell object.
            attr (str): Attribute to retrieve.

        Returns:
            list: List of attribute values for the particle trajectory.
        """
        ret=[]
        particle=self.get_particle(input_ID)
        for cell in particle:
            if isinstance(attributes, str):
                try:
                    ret.append(getattr(cell, attributes))
                except AttributeError:
                    if fill_value is not None:
                        ret.append(fill_value)
                    else:
                        raise AttributeError(f'Cell {cell.n} in frame {cell.frame} does not have attribute {attributes}')
            else:
                cell_attrs=[]
                for attribute in attributes:
                    try:
                        cell_attrs.append(getattr(cell, attribute))
                    except AttributeError:
                        if fill_value is not None:
                            cell_attrs.append(fill_value)
                        else:
                            raise AttributeError(f'Cell {cell.n} in frame {cell.frame} does not have attribute {attribute}')
                ret.append(cell_attrs)
        return ret
    
    def set_particle_attr(self, input_ID, attributes, values):
        """
        Set a specific attribute for a given particle ID or cell.

        Args:
            input_ID (int, np.integer, Cell): ID of the particle or a Cell object.
            attr (str): Attribute to set.
            value (any): Value to set the attribute to.

        Returns:
            list: List of attribute values for the particle trajectory.
        """
        particle=self.get_particle(input_ID)
        if len(particle)!=len(values):
            raise ValueError('Length of values must match the number of cells in the particle')
        for n, cell in enumerate(particle):
            if isinstance(attributes, str):
                setattr(cell, attributes, values[n])
            else:
                for attribute, value in zip(attributes, values[n]):
                    setattr(cell, attribute, value)

    def get_velocities(self, from_csv=False, to_csv=False, **tracking_kwargs):
        """
        Calculate velocities from tracked centroids and optionally save/load from CSV.

        Args:
            from_csv (bool, optional): If True, loads velocities from a CSV file. Defaults to False.
            to_csv (bool, optional): If True, saves velocities to a CSV file. Defaults to False.
            **kwargs: Additional keyword arguments passed to track_centroids function if re-tracking is needed.

        Returns:
            DataFrame: DataFrame containing velocities and related data.
        """
        import os
        import pandas as pd
        import numpy as np
        
        # Load velocities from CSV if requested
        if from_csv:
            try:
                self.velocities = pd.read_csv(self.name.replace('segmented','tracking')[:-1]+'.csv', index_col=0)
                return self.velocities
            except FileNotFoundError:
                print('Couldn\'t find a file at {} to read tracking data from. Retracking.'.format(self.name.replace('segmented','tracking')[:-1]+'.csv'))
        
        # Check if tracking data is available and re-track if necessary
        if hasattr(self, 'tracked_centroids') and tracking_kwargs == {}:
            pass
        else:
            self.track_centroids(**tracking_kwargs)
        
        # Sort a copy of tracking data by particle and frame
        velocities_df = self.tracked_centroids.copy().sort_values(['particle','frame'])
        
        # Calculate displacements
        v = self.tracked_centroids[['x','y','frame','particle']].groupby('particle').apply(np.diff, axis=0, append=np.nan)
        velocities_df[['dx','dy','dt']] = np.concatenate(v)[:,:3]
        velocities_df[['dx','dy']] = velocities_df[['dx','dy']].div(velocities_df['dt'], axis=0)
        
        # Calculate velocity magnitude and direction
        velocities_df['v'] = np.linalg.norm(velocities_df[['dx','dy']], axis=1)
        velocities_df['theta'] = np.arctan2(velocities_df['dy'], velocities_df['dx'])
        
        # Store velocities
        self.velocities = velocities_df
        
        # Save velocities to CSV if requested
        if to_csv:
            print('Writing velocity data to {}.'.format(self.name.replace('segmented','tracking')[:-1]+'.csv'))
            csv_path=Path(self.name.split('/segmented/')[0]+'/tracking/')
            csv_path.makedir(exist_ok=True, parents=True)
            self.velocities.to_csv(self.name.replace('segmented','tracking')[:-1]+'.csv')
        return self.velocities

    def propagate_FUCCI_labels(self):
        '''
        propagates FUCCI data forward in time by copying the last observed value.
        '''
        t=self.tracked_centroids
        for ID in np.unique(t['particle']):
            cell_cycle=self.get_particle_attr(ID, 'cycle_stage', fill_value=0)
            propagated_cell_cycle=np.maximum.accumulate(cell_cycle)
            self.set_particle_attr(ID, 'cycle_stage', propagated_cell_cycle)

    def get_interpretable_FUCCI(self, zero_remainder=True, impute_zeros=True):
        from .FUCCI_linking import get_problematic_IDs, impute_fill, smooth_small_flickers, correct_mother_G2, correct_daughter_G1, remove_sandwich_flicker
        cell_cycle=np.concatenate([frame.fetch_cell_cycle() for frame in self.frames])
        if hasattr(self, 'tracked_centroids') and len(self.tracked_centroids)==len(cell_cycle):
            t=self.tracked_centroids.sort_values(['frame', 'cell_number'])
        else:
            t=self.track_centroids().sort_values(['frame', 'cell_number'])
        t['cell_cycle']=np.concatenate([frame.fetch_cell_cycle() for frame in self.frames])

        for ID in get_problematic_IDs(t): 
            cell_cycle=t.loc[t.particle==ID,'cell_cycle'].copy()

            if impute_zeros:
                # simple fix: extend existing data into any zero regions
                cell_cycle=impute_fill(cell_cycle, limit=6)
            
            # simple fix: smooth single-frame flickers by neighbor values
            cell_cycle=smooth_small_flickers(cell_cycle)

            # manual check: if a cell briefly goes red after G2 and then vanishes, it's just a byproduct of changes before mitosis
            cell_cycle=correct_mother_G2(cell_cycle)

            # manual check: similarly, if a cell starts out orange or red and then goes green, it's just vestigial fluorescence from mitosis.
            cell_cycle=correct_daughter_G1(cell_cycle)
            
            # manual check: if a cell goes from one phase to another and back, it's probably spurious (especially if out of temporal order).
            # the order of these checks matters (especially 3,2,3 then 2,3,2) because lots of cells will flicker and which check goes first determines who wins. Sorted by descending observed frequency
            cell_cycle=remove_sandwich_flicker(cell_cycle, [1,3,1]) # G1 jumps to G2
            cell_cycle=remove_sandwich_flicker(cell_cycle, [3,2,3]) # G2 back to S. This happens all the time because green creeps into red
            cell_cycle=remove_sandwich_flicker(cell_cycle, [1,3,2]) # G1 to S *through* G2. This is also likely because red replaces green
            cell_cycle=remove_sandwich_flicker(cell_cycle, [1,0,2]) # G1 to S through NS. This is also likely because red replaces green

            cell_cycle=remove_sandwich_flicker(cell_cycle, [2,3,2])

            t.loc[t.particle==ID,'cell_cycle']=np.array(cell_cycle)

        if zero_remainder:
            for ID in get_problematic_IDs(t):
                t.loc[t.particle==ID,'cell_cycle']=0 # remove any FUCCI data that still doesn't make sense

        for frame_number, cell_cycle_stages in t.groupby('frame')['cell_cycle']:
            self.frames[frame_number].write_cell_cycle(cell_cycle_stages)

        self.tracked_centroids['cell_cycle']=t['cell_cycle'] # also export cell cycle info to tracking
            
        return t
    #-------------Cell Divisions-------------
    def load_mitoses(self):
        """
        Load mitotic data from a file.
        Reads mitotic data from a file specified by the naming convention of the current data's file name.
        Returns:
            list: List of mitotic events, where each event is represented as a DataFrame containing information about mother and daughter cells.
        """
        
        mitoses_path=str(Path(self.name.replace('segmented','mitoses')).parent)+'.txt'
        mitoses=pd.read_csv(mitoses_path)
        self.mitoses=np.split(mitoses, len(mitoses)/3)
        return self.mitoses
    
    default_weights = (40.79220969,  8.12982495,  0.20812352,  2.54951311, 32.51929981)  # Default weights for mitosis scoring
    default_biases = (0.09863765322259088, 0.11520025039156312, 0.0001280071195560235, 0.00045755121816393185, 0.0015762096995626672)  # Default biases for mitosis scoring

    def get_mitoses(self, persistence_threshold=0, distance_threshold=None, retrack=False, weights=None, biases=None, score_cutoff=1, **kwargs):
        """
        Detect potential mitotic events in the cell tracking data.

        Args:
            persistence_threshold (int, optional): Minimum duration daughter cells must be present after division. Defaults to 0.
            distance_threshold (float, optional): Maximum distance threshold for considering a cell division. If not provided, it is automatically set based on cell area statistics.
            retrack (bool, optional): If True, forces the function to retrack the data. Defaults to False.
            biases (array-like, optional): Biases used in mitosis scoring. Defaults set using manually labeled MDCK data.
            weights (array-like, optional): Weights used in mitosis scoring. Defaults set using manually labeled MDCK data.
            score_cutoff (int, optional): Threshold for considering a detected mitosis event. Defaults to 1.
            **kwargs: Additional keyword arguments to be passed to the tracking function.

        Returns:
            list: List of potential mitotic events, each represented as a DataFrame containing information about mother and daughter cells.
        """
        from scipy.spatial.distance import cdist  # Import cdist function from scipy.spatial.distance
        
        if not distance_threshold:
            # Automatically set maximum search range by cell area statistics
            distance_threshold = 1.5 * np.nanmean([np.sqrt(np.quantile(frame.cell_areas(scaled=False), 0.9)) for frame in self.frames]) 
        
        # Default mitosis scoring parameters
        if not weights:
            weights = self.default_weights
        if not biases:
            biases = self.default_biases
        
        def mitosis_score(params, weights, biases):
            ''' Linear function for scoring mitoses '''
            return np.sum(np.square((np.array(params).T - biases)) * weights, axis=1)

        #norm_means=np.array([0.19036615, 0.23384234, 0.16842723, 0.12717775, 0.18187521])
        #norm_vars=np.array([0.00774153, 0.00920625, 0.08352297, 0.04564848, 0.12634787])
        #linear_weights=np.array([-0.7349965,-0.8832049,-1.0309228,1.0420729,-0.806389])
        #linear_bias=0.3782285
        
        #def mitosis_score(X):
        #    def normalize(X, mean, variance):
        #        return (X-mean)/np.sqrt(variance)

        #    def linear_model(X, weights, bias):
        #        return np.dot([weights], X)+bias

        #    def sigmoid(x):
        #        return 1/(1+np.exp(-x))

        #    return(sigmoid(linear_model(normalize(X, norm_means, norm_vars), linear_weights, linear_bias)).flatten())

        if retrack or not hasattr(self, 'tracked_centroids'):
            # If the data has never been tracked, or if retrack is specified
            trajectories = self.track_centroids(**kwargs)  # Run tracking again
        else:
            trajectories = self.tracked_centroids  # Use existing tracking data

        # Find when cells show up and disappear
        persistent_cells = np.where(trajectories.groupby('particle').size() > persistence_threshold)[0]
        birth_frames = trajectories[trajectories['particle'].isin(persistent_cells)].groupby('particle')['frame'].min()
        death_frames = trajectories.groupby('particle')['frame'].max()
        birth_frames = birth_frames[birth_frames != 0]  # Drop the first frame, when all cells are technically new
        birth_frames = birth_frames.loc[birth_frames.duplicated(keep=False)]  # Drop frames which only have one new cell show up (need two to be a plausible cell division)

        frames_of_interest = pd.Series(np.intersect1d(birth_frames.values, death_frames.values + 1))
        trajectories = trajectories.set_index(['particle', 'frame'])  # Make tracking data indexable by tracking info for fast identification
        
        # Look for mitoses out of potential candidates
        mitoses = []
        for frame_number in frames_of_interest:
            # Get relevant particle IDs for this frame
            birth_IDs = birth_frames[birth_frames == frame_number].index.values
            death_IDs = death_frames[death_frames == frame_number - 1].index.values

            births = trajectories.loc[zip(birth_IDs, [frame_number] * len(birth_IDs))].reset_index()
            deaths = trajectories.loc[zip(death_IDs, [frame_number - 1] * len(death_IDs))].reset_index()

            mother_candidates, daughter_candidates = np.where(cdist(deaths[['x', 'y']], births[['x', 'y']]) < distance_threshold)
            mitosis_candidates = births.iloc[daughter_candidates].copy()
            mitosis_candidates['mother_candidates'] = mother_candidates
            mitosis_candidates = mitosis_candidates[mitosis_candidates['mother_candidates'].duplicated(keep=False)]

            for mother_candidate_ID in np.unique(mitosis_candidates['mother_candidates']):
                # Parameters: m_circ, d_circ, d_distance, d_angle, d_CoM, area_ratio, area_change
                potential_daughter_IDs = mitosis_candidates[mitosis_candidates.mother_candidates == mother_candidate_ID]
                potential_mother_ID = deaths.iloc[[mother_candidate_ID]]  # Get the mother cell
                mother_cell = self.frames[potential_mother_ID['frame'].item()].cells[potential_mother_ID['cell_number'].item()]
                
                # Potential mother attributes
                m_size = np.sqrt(mother_cell.area())
                extended_mother = trajectories.loc[potential_mother_ID['particle']].iloc[-3:].reset_index()
                m_circ = 1 - np.max([self.frames[frame_number].cells[cell_number].circularity for frame_number, cell_number in np.array(extended_mother[['frame', 'cell_number']])])
                
                # Potential daughter attributes
                d_circs = []
                d_areas = []
                for frame_number, cell_number in np.array(potential_daughter_IDs[['frame', 'cell_number']]):
                    daughter_cell = self.frames[frame_number].cells[cell_number]
                    d_circs.append(daughter_cell.circularity)
                    d_areas.append(daughter_cell.area())
                d_circs = 1 - np.array(d_circs)
                d_areas = np.array(d_areas)

                d_displacements = potential_daughter_IDs[['x', 'y']] - potential_mother_ID[['x', 'y']].values
                d_distances = np.linalg.norm(d_displacements, axis=1)
                d_angles = np.arctan2(d_displacements['y'], d_displacements['x']).to_numpy()
                
                # Score daughter pairs
                all_pairs = np.stack(np.triu_indices(len(potential_daughter_IDs), k=1), axis=-1)

                d_angle = np.abs(np.diff(d_angles[all_pairs] / np.pi, axis=1).flatten() % 2 - 1)
                d_circ = np.min(d_circs[all_pairs], axis=1)
                d_CoM = np.sum(np.array(potential_daughter_IDs[['y', 'x']])[all_pairs] * d_areas[all_pairs].reshape(-1, 2, 1), axis=1) / np.sum(d_areas[all_pairs], axis=1).reshape(-1, 1)
                CoM_displacement = np.linalg.norm(d_CoM - potential_mother_ID[['y', 'x']].values, axis=1) / m_size
                d_distance_diff = np.diff(d_distances[all_pairs]).flatten() / m_size
                m_circs = np.ones(len(all_pairs)) * m_circ
                
                # Calculate overall score and save if it passes the cutoff
                mitosis_scores = mitosis_score([m_circs, d_circ, d_distance_diff, d_angle, CoM_displacement], weights, biases)
                best_score = np.argmin(mitosis_scores)
                if mitosis_scores[best_score] < score_cutoff:
                    m = pd.concat([potential_mother_ID, potential_daughter_IDs.iloc[all_pairs[best_score]].drop('mother_candidates', axis=1)])
                    mitoses.append(m.set_index(['particle', 'frame']))

        self.mitoses = mitoses
        return self.mitoses
    
     # -------------Indexing Lower Objects-------------
    def densities(self):
        '''returns an array of densities for each frame in chronological order'''
        return np.array([frame.mean_density() for frame in self.frames])
        
    def centroids(self):
        '''returns a DataFrame of cell centroids, with additional columns specifying the cell and frame numbers. Used for particle tracking.'''
        all_centroids=[]
        for frame_number, frame in enumerate(self.frames): # iterate through frames
            centroids=pd.DataFrame(frame.centroids(), columns=['y','x']) # get centroids
            centroids['frame']=frame_number # label each set of centroids with their frame number
            all_centroids.append(centroids)
        all_centroids=pd.concat(all_centroids).rename_axis('cell_number').reset_index()
        return all_centroids
    
    def shape_parameters(self):
        '''returns a DataFrame of shape parameters, with additional columns specifying the cell and frame numbers. Only returns cells with a full set of neighbors'''
        all_q=[]
        for frame_number, frame in enumerate(self.frames): # iterate through frames
            q=pd.DataFrame(frame.shape_parameters(), columns=['q',]) # get shape parameter
            q['frame']=frame_number # label each set of shape parameters with their frame number
            all_q.append(q)
        all_q=pd.concat(all_q).rename_axis('cell_number').reset_index()
        return all_q
    
    def load_img(self, **kwargs):
        for frame in self.frames:
            frame.load_img(**kwargs)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

class Image:
    ''' takes one segmented image as a seg.npy file path and runs the numbers. '''
    verbose=False
    def vprint(self, obj):
        if self.verbose:
            print(obj)
        else:
            pass

    def __init__(self, file_path, frame_number=None, mend=False, max_gap_size=300, verbose=False, overwrite=False, load_img=False, normalize=False, scale=0.3225, units='microns'):
        '''
        takes a file path and loads the data.
        
        verbose=True makes the Image print out updates on what it's doing.
        mend=True checks for and closes any pixel-sized holes in the monolayer. (~3x slower).
        max_gapsize (int) is the upper limit on contiguous empty pixels that will be mended with nearby data.
        overwrite=True rewrites the seg.npy at file_path with whatever processing has been applied.
        load_img=True loads the image into the img attribute. Defaults to False to save memory.
        Normalize=True sets the maximum img value to 1.
        '''
        import cellpose.utils as cp_utils

        # Set pixel units and scale
        self.units = units  # Units for measurements
        self.scale = scale  # Scale factor for converting pixels to desired units
        self.frame_number = frame_number  # Frame number for the image sequence

        # Print debug statements if verbose mode is enabled
        self.verbose = verbose

        # Print loading information
        self.vprint('loading {}'.format(file_path))

        # Load data from file
        data = np.load(file_path, allow_pickle=True).item()

        # Fetch data and metadata from seg.npy
        self.name = file_path  # File name
        self.n = np.max(data['masks'])  # Number of detected cells in field of view (FOV)
        
        # Load image if specified or for overwriting
        if load_img or overwrite:
            self.img = data['img']  # Load image data
            if normalize:
                self.img = preprocessing.normalize(self.img, dtype=np.float32)  # Normalize image data if specified
        
        # Load masks and set resolution
        self.masks = data['masks']  # Masks for cell detection
        
        # set masks datatype
        if self.masks.max() < 65535: # there will always be fewer than 65535 cells in a single frame, just a failsafe
            self.masks = self.masks.astype(np.uint16)
        else:
            self.masks = self.masks.astype(np.uint32)

        self.resolution = self.masks.shape  # Image resolution

        # Check for and mend gaps in masks
        if mend:
            self.masks, mended = preprocessing.mend_gaps(self.masks, max_gap_size)
        else:
            mended = False
        
        # Generate outlines channel or load from file
        if mended or 'outlines' not in data.keys():
            self.outlines = cp_utils.masks_to_outlines(self.masks)  # Generate outlines from masks
        else:
            self.outlines = data['outlines']  # Load outlines from data file
        
        try:
            self.outlines=self.outlines.todense()
        except AttributeError: # if it's already dense, carry on
            pass
        self.outlines=self.outlines!=0
        
        # Generate outlines_list or load from file
        if mended or 'outlines_list' not in data.keys():
            self.vprint('creating new outlines_list from masks')
            outlines_list = cp_utils.outlines_list_multi(self.masks)  # Generate outlines_list
        else:
            outlines_list = data['outlines_list']  # Load outlines_list from file
        
        # Set additional attributes from data file
        for attr in set(data.keys()).difference(['img', 'masks', 'outlines', 'outlines_list']):
            setattr(self, attr, data[attr])  # Set additional attributes
        
        
        # Overwrite file if specified
        if overwrite:
            export = {'img': data['img'], 'masks': self.masks, 'outlines': self.outlines, 'outlines_list': outlines_list}
            if hasattr(self, 'FUCCI'):  # Export FUCCI channels if present
                export['FUCCI'] = self.FUCCI
            
            # Export the data and overwrite the segmentation file
            self.vprint('overwriting old file at {}'.format(file_path))
            np.save(file_path, export)
        
        # Instantiate Cell objects for each cell labeled in the image
        self.cells = np.array([Cell(n, outlines_list[n], frame_number=frame_number) for n in range(self.n)])
        
        if hasattr(self, 'cell_cycles'):
            self.write_cell_cycle(self.cell_cycles)

    
     # -------------Image Processing-------------    
    def load_img(self, normalize=False):
        self.img=np.load(self.name, allow_pickle=True).item()['img']
        if normalize:
            self.img=preprocessing.normalize(self.img, dtype=np.float32) # normalize

    def to_seg_npy(self, export_path=None, overwrite_img=False, write_attrs=[]):
        data=np.load(self.name, allow_pickle=True).item()
        outlines_list=[cell.outline for cell in self.cells]

        if not overwrite_img:
            img=data['img']
        else:
            img=self.img

            
        export={'img':img, 'masks':self.masks, 'outlines':self.outlines, 'outlines_list':outlines_list}
        for attr in write_attrs:
            export[attr]=getattr(self, attr)

        if hasattr(self, 'FUCCI'): # export FUCCI channels if present
            export['FUCCI']=self.FUCCI

        if export_path is None: # if no export path is given, overwrite existing file
            export_path=self.name

        Path(export_path).parent.mkdir(parents=True, exist_ok=True) # make sure the directory exists
        np.save(export_path, export) # write segmentation file
    
    def delete_cell(self, cell_number):
        '''deletes a cell from the image.'''
        self.cells=np.delete(self.cells, cell_number)

        self.masks=preprocessing.renumber_masks(self.masks)
        self.n=self.masks.max()

        for n, cell in enumerate(self.cells):
            cell.n=n
    # ------------FUCCI----------------        
    def measure_FUCCI(self, percent_threshold=0.15, red_fluor_threshold=None, green_fluor_threshold=None, orange_brightness=1.5, threshold_offset=0, noise_score=0.02):
        """
        Measure the FUCCI (Fluorescence Ubiquitination Cell Cycle Indicator) levels in cells.

        Args:
            use_masks (bool): If True, use masks for measurements.
            percent_threshold (float): Threshold for percentage comparison.
            red_fluor_threshold (float): Threshold for red fluorescence.
            green_fluor_threshold (float): Threshold for green fluorescence.

        Returns:
            tuple: Red fluorescence percentages, Green fluorescence percentages.
        """
        # Ensure FUCCI data is available
        if not hasattr(self, 'FUCCI'):  # Check if FUCCI data exists
            raise AttributeError('use FUCCI_preprocess.ipynb to generate FUCCI data first')
        nuclear_size_threshold=np.median(self.cell_areas(scaled=False))*percent_threshold
        red, green = self.FUCCI
        
        # threshold fluorescence levels
        if not red_fluor_threshold:
            red_fluor_threshold=preprocessing.get_fluor_threshold(red, nuclear_size_threshold, noise_score=noise_score)+threshold_offset
        if not green_fluor_threshold:
            green_fluor_threshold=preprocessing.get_fluor_threshold(green, nuclear_size_threshold, noise_score=noise_score)+threshold_offset
        self.red_fluor_threshold=red_fluor_threshold
        self.green_fluor_threshold=green_fluor_threshold
        self.orange_brightness=orange_brightness
        thresholded_green=green>green_fluor_threshold
        thresholded_red=red>red_fluor_threshold
        thresholded_orange=(red>red_fluor_threshold*orange_brightness)&thresholded_green

        fluor_percentages=np.stack([preprocessing.fluorescent_percentages(self.masks, thresholded_green),
                                    preprocessing.fluorescent_percentages(self.masks, thresholded_red),
                                    preprocessing.fluorescent_percentages(self.masks, thresholded_orange)], axis=1)

        fluor_nuclei=fluor_percentages>percent_threshold
        fluor_nuclei[fluor_nuclei[:,2], :2]=0 # wherever cells are orange, turn off red and green

        red_or_green=fluor_nuclei[:,0]&fluor_nuclei[:,1]
        fluor_nuclei[red_or_green, np.argmin(fluor_percentages[red_or_green,:2], axis=1)]=0 # if not orange, pick red or green based on which has a higher percentage
        cc_stage_number=np.argmax(fluor_nuclei, axis=1)+1 # G1->1, S->2, G2->3
        cc_stage_number[np.sum(fluor_nuclei, axis=1)==0]=0 # no signal->0
        for cell, cycle_stage in zip(self.cells, cc_stage_number):
            cell.cycle_stage=cycle_stage
        return cc_stage_number
    
    def fetch_cell_cycle(self):
        return self.get_cell_attr('cycle_stage')

    def write_cell_cycle(self, cell_cycle_stages):
        self.set_cell_attr('cycle_stage', cell_cycle_stages)
    
    # -------------Metrics---------------
    # generalized methods for getting and setting cell attributes
    def get_cell_attr(self, attributes, fill_value=None):
        ret=[]
        for cell in self.cells:
            if isinstance(attributes, str):
                try:
                    ret.append(getattr(cell, attributes))
                except AttributeError:
                    if fill_value is not None:
                        ret.append(fill_value)
                    else:
                        raise AttributeError(f'Cell {cell.n} does not have attribute {attributes}')
            else:
                cell_attrs=[]
                for attribute in attributes:
                    try:
                        cell_attrs.append(getattr(cell, attribute))
                    except AttributeError:
                        if fill_value is not None:
                            cell_attrs.append(fill_value)
                        else:
                            raise AttributeError(f'Cell {cell.n} does not have attribute {attribute}')
                ret.append(cell_attrs)
        return ret

    def set_cell_attr(self, attributes, values):
        for cell, value in zip(self.cells, values):
            if isinstance(attributes, str): # if only one attribute is being set
                setattr(cell, attributes, value)
            else: # if multiple attributes are being set
                for attribute, val in zip(attributes, value):
                    setattr(cell, attribute, val)

    def cell_areas(self, scaled=True):
        """
        Calculate areas of individual cells in the image (area given as number of pixels).
        Args:
            scaled (bool, optional): If True, returns scaled areas considering image scaling. Defaults to True.
        Returns:
            numpy.ndarray: Array of cell areas.
        """
        # Find unique cell labels and their counts (n_pixels)
        areas = np.unique(self.masks, return_index=False, return_counts=True)[1][1:]
        
        if scaled:  # If scaled option is True, scale the areas based on image scaling factor
            return areas * self.scale ** 2
        else:
            return areas  # Otherwise, return the unscaled areas

    def mean_cell_area(self, scaled=True):
        '''get cell areas by counting number of pixels per mask'''
        mean_area=np.mean(self.cell_areas(scaled))
        return mean_area
    
    def FOV_area(self, scaled=True):
        '''width*height drawn from resolution'''
        FOV_pixels=np.multiply(*self.resolution)
        if scaled:
            return FOV_pixels*self.scale**2
        else:
            return FOV_pixels
        
    def mean_cell_length(self, scaled=True):
        '''characteristic length scale for particle tracking and mitosis detection'''
        return np.sqrt(self.mean_cell_area(scaled))

    def mean_density(self):
        '''cells per square mm, measured by taking inverse of mean cell area'''
        return 1e6/self.mean_cell_area() # (1e6 mm^2/um^2)/(um^2 per cell)=cells per mm^2
    
    # -------------Centroids-------------
    def centroids(self):
        '''returns an array of all cell centroids.'''
        if len(self.cells)==0:
            return np.empty((0,2)) # no cells in this frame
        
        elif not hasattr(self.cells[-1], 'centroid'): # check if centroids have been calculated just by checking the last cell
            self.get_centroids()
        
        return np.array(self.get_cell_attr('centroid'))
    
    def get_centroids(self):
        '''adds centroid attribute to each Cell in the Image.'''
        centroids=np.array(ndimage.center_of_mass(np.ones(self.resolution),self.masks,np.arange(1,self.n+1)))
        self.set_cell_attr('centroid', centroids)
    
    # -------------Fit Ellipses to Cells-------------
    def fit_ellipses(self):
        '''fits each cell outline to an ellipse, and assigns the angle and aspect ratio data to the corresponding Cell.'''
        for cell in self.cells: # can I vectorize fitting all these ellipses? I just don't really know how this works well enough to optimize I guess
            cell.fit_ellipse() 
    
    def cell_ellipse_parameters(self):
        '''fetches ellipse fitting data from each Cell and returns a DataFrame for all Cells in the Image.'''
        try: # see if all the theta and aspect ratio data is there to be spit out
            return pd.DataFrame([[cell.theta,cell.aspect] for cell in self.cells], columns=['theta','aspect'])
        
        except AttributeError: # looks like at least some of it's missing, so generate it
            self.fit_ellipses()
            df=[]
            for cell in self.cells:
                if hasattr(cell, 'theta') and hasattr(cell, 'aspect'):
                    df.append([cell.theta,cell.aspect])
            return pd.DataFrame(df, columns=['theta','aspect'])

    # -------------Vertex Reconstruction-------------
    def get_TCJs(self):
        '''set Cell.vertices attribute for all cells with complete set of neighbors.'''
        from scipy.signal import convolve2d
        # find good cells
        self.bad_cell_indices=np.unique(self.masks[np.where(convolve2d(self.masks!=0,[[0,1,0],[1,1,1],[0,1,0]])[1:-1,1:-1]<5)])[1:]-1 # indices of cells without full set of neighbors
        self.good_cell_indices=np.delete(np.arange(self.n),self.bad_cell_indices) # indices of cells with full set of neighbors (can reconstruct vertices from TCJs)
        
        if len(self.cells)==0:
            self.TCJs=np.empty((0,2))
            return self.TCJs
        
        elif not hasattr(self.cells[0], 'centroid'): # get centroids if they're not already found
            self.get_centroids() # need centroids to order vertices

        # get bottom-right corners of every non-zero square of pixels
        cc_junctions=np.column_stack(np.where(convolve2d(self.outlines,np.array([[1,1],[1,1]]))==4))
        
        # find 2x2 boxes with three or four cells in them (our metric for TCJs)
        is_TCJ=lambda x,y: len(np.unique(self.masks[x-1:x+1,y-1:y+1]))>=3 # very rare cases will have four-cell junctions, generally three
        is_TCJ_vect=np.vectorize(is_TCJ)
        TCJs=cc_junctions[is_TCJ_vect(cc_junctions[:,0],cc_junctions[:,1])]
        self.TCJs=TCJs
        
        # figure out which cells participate in each TCJ
        TCJ_cells=[]
        for x,y in TCJs:
            TCJ_cells.append(self.masks[x-1:x+1,y-1:y+1].flatten()) # kind of silly to do this again but idk how to vectorize and return a more complex output than True/False
        TCJ_cells=np.array(TCJ_cells)-1 # subtract 1 so cells are 0-indexed
        self.TCJ_cells=TCJ_cells
        
        # sort TCJs into their constituent Cell objects
        incomplete_polygons=[]
        for good_cell in [self.cells[n] for n in self.good_cell_indices]:
            good_cell.vertices=np.unique(TCJs[np.where(TCJ_cells==good_cell.n)[0]],axis=0) # find full set of vertices for the nth cell
            good_cell.sorted_vertices=good_cell.sort_vertices() # get the order in which vertices are connected
            if len(good_cell.vertices)<3: # find cells with 2 or fewer found vertices (line or point, not a polygon. Probably something went wrong)
                incomplete_polygons.append(good_cell.n)
                self.vprint('found an incomplete polygon at index {} of {}'.format(good_cell.n, self.name))
        if len(incomplete_polygons)>0:
            self.good_cell_indices=self.good_cell_indices[~np.any([self.good_cell_indices==x for x in incomplete_polygons], axis=0)] # drop incomplete polygons
            self.bad_cell_indices=np.append(self.bad_cell_indices,incomplete_polygons) # for consistency, add incomplete polygons to our index of bad cells. Not using this rn.
        return self.TCJs

    def good_cells(self):
        '''returns a list of the Cell objects with a full set of neighbors.'''
        if not hasattr(self, 'good_cell_indices'):
            self.get_TCJs()
        return self.cells[self.good_cell_indices]
    
    def shape_parameters(self):
        '''returns an array of all shape parameters.'''
        shape_parameters=[]
        for cell in self.good_cells():
            shape_parameters.append(cell.shape_parameter)
        return np.array(shape_parameters)

    def cell_polygons(self,ec='k',fc='lightsteelblue',linewidths=0.8,**kwargs):
        '''returns a matplotlib PatchCollection of vertex-reconstructed cells for efficient plotting.'''
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        cell_polygons=[]
        for cell in self.good_cells():
            vertices=cell.sorted_vertices
            vertices=np.array([vertices[:,1],vertices[:,0]]).T # flip x and y for plotting
            cell_polygons.append(Polygon(vertices))
        cell_collection=PatchCollection(cell_polygons, edgecolor=ec,facecolor=fc,linewidths=linewidths,**kwargs)
        return cell_collection
    
class HeightMap(Image):
    def __init__(self, seg_path, mesh_path=None, zero_to_nan=True, scale=0.1625, z_scale=0.3225, NORI=False, **kwargs):
        if mesh_path is None:
            mesh_path=seg_path.replace('segmented','heights').replace('seg.npy', 'binarized.tif')
        self.z, self.height_img=preprocessing.read_height_tif(mesh_path, z_scale=z_scale, zero_to_nan=zero_to_nan)
        super().__init__(seg_path, scale=scale, **kwargs)

        self.masks_3d=np.repeat(self.masks[np.newaxis], self.height_img.shape[0], axis=0)
        self.masks_3d[~self.height_img]=0

        if NORI:
            self.read_NORI()
    
    def read_NORI(self, file_path=None, mask_nan_z=True):
        from skimage import io
        if file_path is None:
            file_path=self.name.replace('segmented','NORI').replace('seg.npy', 'NORI.tif')
        self.NORI=io.imread(file_path).astype(float)
        if mask_nan_z:
            #np.ma.masked_where(np.repeat(np.isnan(self.z)[np.newaxis], self.NORI.shape[0], axis=0), self.NORI, copy=False)
            self.NORI[:, np.isnan(self.z)]=np.nan
        return self.NORI
    
    def get_NORI_density(self):
        if not hasattr(self, 'NORI'):
            self.read_NORI()
        self.NORI_density=np.array([ndimage.mean(self.NORI[...,i], labels=self.masks_3d, index=range(1,self.masks.max()+1)) for i in range(3)])
        self.NORI_density_std=np.array([ndimage.standard_deviation(self.NORI[...,i], labels=self.masks_3d, index=range(1,self.masks.max()+1)) for i in range(3)])
        self.set_cell_attr('NORI_density', self.NORI_density.T)
        return self.NORI_density
    
    def get_NORI_mass(self):
        if not hasattr(self, 'NORI'):
            self.read_NORI()
        self.NORI_mass=np.array([ndimage.sum(self.NORI[...,i], labels=self.masks_3d, index=range(1,self.masks.max()+1)) for i in range(3)])
        self.set_cell_attr('NORI_mass', self.NORI_mass.T)
        return self.NORI_mass
    
    
    def get_volumes(self):
        #self.heights=self.get_heights()
        self.volumes=ndimage.sum(self.z, labels=self.masks, index=range(1,self.masks.max()+1))*self.scale**2
        self.mean_heights=ndimage.mean(self.z, labels=self.masks, index=range(1,self.masks.max()+1))
        self.set_cell_attr('volume', self.volumes)
        self.set_cell_attr('height', self.mean_heights)

        return self.volumes

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

class Cell:
    '''class for each labeled cell membrane.'''
    def __init__(self, n, outline, frame_number=None, **kwargs):
        self.frame=frame_number
        self.n=n
        self.outline=outline

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def area(self):
        area=0.5*np.abs(np.dot(self.outline.T[0],np.roll(self.outline.T[1],1))-np.dot(self.outline.T[1],np.roll(self.outline.T[0],1)))
        return area
    
    @property
    def perimeter(self):
        perimeter=np.sum(np.linalg.norm(np.diff(self.outline, axis=0, append=[self.outline[0]]).T, axis=0))
        return perimeter
    
    @property
    def circularity(self):
        circularity=4*np.pi*self.area/self.perimeter**2
        return circularity
    
    def sort_vertices(self):
        '''
        determines which vertices are connected by ordering polar angles to each vertex w.r.t. the centroid.
        some edge cases where this won't work (unusual concave structures, radial line segments) but I think these are sufficiently unlikely in physiological cells.
        could replace this with a sort_vertices which pulls them by outline now that I have that.
        only calculable for cells with reconstructed vertices ('good cells').
        '''
        zeroed_coords=self.vertices-self.centroid # zero vertices to the centroid
        angles=np.arctan2(zeroed_coords[:,0],zeroed_coords[:,1]) # get polar angles
        vertex_order=np.argsort(angles) # sort polar angles
        return self.vertices[vertex_order]

    def TCJ_axis(self):
        """
        Compute the axis orientation based on the spatial distribution of tricellular junctions (TCJs).

        Returns:
            float or bool: Orientation angle of the axis if computation is successful, False otherwise.
        """

        try:
            TCJs = np.flip(self.sorted_vertices, axis=1)
        except AttributeError:
            # If the cell doesn't have a full set of neighbors or get_TCJs() hasn't been run
            return False
        
        # Calculate the center of mass of TCJs
        CoM_J = TCJs.mean(axis=0)

        # Calculate the scatter matrix
        a = [np.outer(junction - CoM_J, junction - CoM_J) for junction in TCJs]
        S = np.mean(a, axis=0)

        # Compute eigenvalues and eigenvectors of the scatter matrix
        l, eigen = np.linalg.eig(S)

        # Determine the major axis and calculate its angle
        shapeAxis = eigen.T[np.argmax(l)]
        self.circ_J = np.min([l[0] / l[1], l[1] / l[0]])
        self.theta_J = np.arctan2(*np.flip(shapeAxis))

        return self.theta_J
    
    def perimeter_axis(self):
        """
        Compute the axis orientation based on the perimeter of the cell.

        Returns:
            float: Orientation angle of the axis.
        """
        vertices = self.outline
        
        # Calculate segment lengths
        segment_lengths = np.linalg.norm(np.diff(vertices, append=[vertices[0]], axis=0), axis=1)
        
        # Calculate center of mass of the perimeter
        CoM_P = np.sum([1 / 2 * (vertices[i] + vertices[(i + 1) % len(vertices)]) * segment_lengths[i] for i in range(len(vertices))], axis=0) / self.perimeter
        
        # Translate vertices to center of mass
        zeroed_vertices = vertices - CoM_P
        
        # Calculate scatter matrix
        summation = []
        for i in range(len(vertices)):
            v_current = zeroed_vertices[i]
            v_next = zeroed_vertices[(i + 1) % len(vertices)]
            summation.append(segment_lengths[i] * ((np.outer(v_next, v_next) + np.outer(v_current, v_current)) / 3 + (np.outer(v_next, v_current) + np.outer(v_current, v_next)) / 6))
        S_P = np.sum(summation, axis=0)
        
        # Compute eigenvalues and eigenvectors of the scatter matrix
        l, eigen = np.linalg.eig(S_P)
        
        # Determine the major axis and calculate its angle
        shapeAxis = eigen.T[np.argmax(l)]
        self.circ_P = np.min([l[0] / l[1], l[1] / l[0]])
        self.theta_P = np.arctan2(*np.flip(shapeAxis))
        
        return self.theta_P

    def poly_perimeter_axis(self):
        """
        Compute the axis orientation based on the perimeter of the polygon formed by connecting TCJs.

        Returns:
            float or bool: Orientation angle of the axis if computation is successful, False otherwise.
        """
        try:
            TCJs = np.flip(self.sorted_vertices, axis=1)
        except AttributeError:
            # If the cell doesn't have a full set of neighbors or get_TCJs() hasn't been run
            return False
        
        # Calculate segment lengths of the polygon
        poly_segment_lengths = np.linalg.norm(np.diff(TCJs, append=[TCJs[0]], axis=0), axis=1)
        poly_perimeter = poly_segment_lengths.sum()
        
        # Calculate center of mass of the polygon perimeter
        CoM_P = np.sum([1 / 2 * (TCJs[i] + TCJs[(i + 1) % len(TCJs)]) * poly_segment_lengths[i] for i in range(len(TCJs))], axis=0) / poly_perimeter
        
        # Translate TCJs to center of mass
        zeroed_TCJs = TCJs - CoM_P
        
        # Calculate scatter matrix
        summation = []
        for i in range(len(TCJs)):
            v_current = zeroed_TCJs[i]
            v_next = zeroed_TCJs[(i + 1) % len(TCJs)]
            summation.append(poly_segment_lengths[i] * ((np.outer(v_next, v_next) + np.outer(v_current, v_current)) / 3 + (np.outer(v_next, v_current) + np.outer(v_current, v_next)) / 6))
        S_P = np.sum(summation, axis=0)
        
        # Compute eigenvalues and eigenvectors of the scatter matrix
        l, eigen = np.linalg.eig(S_P)
        
        # Determine the major axis and calculate its angle
        shapeAxis = eigen.T[np.argmax(l)]
        self.circ_polyP = np.min([l[0] / l[1], l[1] / l[0]])
        self.theta_polyP = np.arctan2(*np.flip(shapeAxis))
        
        return self.theta_polyP

    @property
    def shape_parameter(self):
        '''
        calculates the shape parameter q=perimeter/area**2 using vertex data. 
        only calculable for cells with reconstructed vertices ('good cells').
        '''
        shape_parameter=self.vertex_perimeter/(self.vertex_area**0.5)
        return shape_parameter
    
    @property
    def vertex_area(self):
        '''returns the area of the cell as calculated from the vertices.'''
        y,x=self.sorted_vertices.T
        vertex_area=0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        return vertex_area
    
    @property
    def vertex_perimeter(self):
        '''returns the perimeter of the cell as calculated from the vertices.'''
        vertex_perimeter=np.sum(np.linalg.norm(np.diff(self.sorted_vertices, append=[self.sorted_vertices[0]], axis=0), axis=1))
        return vertex_perimeter
    
    def fit_ellipse(self):
        '''
        uses skimage's EllipseModel to fit an ellipse to the outline of the cell.
        '''
        from skimage.measure import EllipseModel
        ellipse_model=EllipseModel()
        
        ellipse_model.estimate(self.outline)
        if ellipse_model.params:
            a,b,theta=ellipse_model.params[2:]
        else:
            return False
        
        self.fit_params=ellipse_model.params # mildly redundant but preserve all the fit parameters for more convenient plotting
        
        if a>b:
            self.aspect=a/b
            self.theta=theta%np.pi
        else:
            self.aspect=b/a
            self.theta=(theta+np.pi/2)%np.pi
            
        return self.fit_params