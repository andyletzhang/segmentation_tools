import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from itertools import combinations
from segmentation_tools.networks import min_weighted_independent_set

# Default weights for mitosis scoring
default_weights = (40.79220969, 8.12982495, 2.0, 0.20812352, 2.54951311, 32.51929981)
# Default biases for mitosis scoring
default_biases = (0.1, 0.1, 0, 0, 0, 0)


def get_viable_mitoses(stack, distance_threshold=1.5, persistence_threshold=0):
    """
    Detect mitosis events in a stack of segmented frames.
    """

    mitosis_candidates = get_division_candidates(stack, distance_threshold, persistence_threshold)
    mitosis_df = pd.concat(
        [evaluate_mitosis(stack, mother, daughters) for mother, daughters in mitosis_candidates.items()], ignore_index=True
    )
    return mitosis_df


def get_mitosis_scores(mitosis_df, weights=[1, 1, 1, 1, 1, 1], biases=None):
    weights = np.array(weights) * default_weights
    if biases is None:
        biases = default_biases
    mitosis_df['score'] = mitosis_score(
        mitosis_df[['mother_circ', 'daughter_circ', 'daughter_distance', 'distance_diff', 'angle', 'CoM_displacement']].values,
        weights=weights,
        biases=biases,
    ).sum(axis=1)

    return mitosis_df


def threshold_mitoses(mitosis_df, threshold=1):
    mitosis_df = mitosis_df[mitosis_df['score'] < threshold]
    return resolve_conflicts(mitosis_df)


def get_division_candidates(stack, distance_threshold=1.5, persistence_threshold=0):
    """
    Detect potential cell division events from particle trajectories.

    Args:
        distance_threshold: Maximum distance threshold for daughter cells (multiplied by sqrt of 90th percentile cell area)
        persistence_threshold: Minimum number of frames a cell must persist

    Returns:
        DataFrame of mitosis candidates with columns ['frame', 'mother', 'daughter1', 'daughter2']
    """
    # Calculate adaptive distance threshold based on cell areas
    area_threshold = np.nanmean([np.sqrt(np.quantile(frame.cell_areas(scaled=False), 0.9)) for frame in stack.frames])
    distance_threshold *= area_threshold

    trajectories = stack.tracked_centroids
    # Filter for persistent cells and get birth/death frames
    cell_counts = trajectories.groupby('particle').size()
    persistent_cells = cell_counts[cell_counts > persistence_threshold].index

    # Create efficient lookup for birth and death frames
    frame_data = (
        trajectories[trajectories['particle'].isin(persistent_cells)].groupby('particle').agg({'frame': ['min', 'max']}).frame
    )

    birth_frames = frame_data['min']
    death_frames = frame_data['max']

    # Filter birth frames
    valid_birth_frames = birth_frames[
        (birth_frames != 0)  # Drop first frame
        & birth_frames.duplicated(keep=False)  # Keep only frames with multiple births
    ]

    # Find frames where deaths and subsequent births occur
    frames_of_interest = pd.Series(np.intersect1d(valid_birth_frames.values, death_frames.values + 1)).sort_values()

    # Pre-compute trajectory lookup
    traj_lookup = trajectories.set_index(['particle', 'frame'])

    mitosis_candidates = {}

    for frame_number in frames_of_interest:
        # Get births and deaths for current frame
        birth_mask = birth_frames == frame_number
        death_mask = death_frames == frame_number - 1

        birth_IDs = birth_frames[birth_mask].index
        death_IDs = death_frames[death_mask].index

        if len(birth_IDs) < 2 or len(death_IDs) == 0:
            continue

        # Get coordinates
        births = traj_lookup.loc[zip(birth_IDs, [frame_number] * len(birth_IDs))][['x', 'y']]
        deaths = traj_lookup.loc[zip(death_IDs, [frame_number - 1] * len(death_IDs))][['x', 'y']]

        # Calculate distances and find valid pairs
        distances = cdist(deaths, births)
        mother_idx, daughter_idx = np.where(distances < distance_threshold)

        if len(mother_idx) == 0:
            continue

        # Find mothers with exactly 2 or more daughters
        mother_counts = pd.Series(mother_idx).value_counts()
        valid_mothers = mother_counts[mother_counts >= 2].index
        # Generate candidate pairs
        for mother in valid_mothers:
            daughters = daughter_idx[mother_idx == mother]
            mitosis_candidates[death_IDs[mother]] = np.array(birth_IDs[daughters])

    return mitosis_candidates


def mitosis_score(params, weights=default_weights, biases=default_biases):
    """Linear function for scoring mitoses"""
    return np.square((np.array(params) - biases)) * weights


def evaluate_mitosis(stack, mother, daughters):
    mother_cell = stack.get_particle(mother)
    mother_size = np.sqrt(mother_cell[0].area_pixels)
    mother_circ = 1 - np.max([m.circularity for m in mother_cell[-3:]])
    mother_centroid = mother_cell[-1].corrected_centroid

    daughter_circs = []
    daughter_centroids = []
    daughter_areas = []
    for daughter in daughters:
        daughter_cell = stack.get_particle(daughter)
        daughter_circ = 1 - np.max([m.circularity for m in daughter_cell[:2]])
        daughter_centroids.append(daughter_cell[0].corrected_centroid)
        daughter_circs.append(daughter_circ)
        daughter_areas.append(daughter_cell[0].area_pixels)
    daughter_centroids = np.array(daughter_centroids)
    daughter_areas = np.array(daughter_areas)
    daughter_circs = np.array(daughter_circs)
    daughter_distances = cdist([mother_centroid], daughter_centroids)[0]
    daughter_angles = np.arctan2(daughter_centroids[:, 1] - mother_centroid[1], daughter_centroids[:, 0] - mother_centroid[0])

    daughter_pairs = list(combinations(range(len(daughters)), 2))
    pair_CoM = np.sum(daughter_centroids[daughter_pairs] * daughter_areas[daughter_pairs].reshape(-1, 2, 1), axis=1) / np.sum(
        daughter_areas[daughter_pairs], axis=1
    ).reshape(-1, 1)

    pair_angle = np.abs(np.diff(daughter_angles[daughter_pairs] / np.pi, axis=1).flatten() % 2 - 1)
    pair_circ = np.mean(daughter_circs[daughter_pairs], axis=1)
    pair_CoM_displacement = np.linalg.norm(pair_CoM - mother_centroid, axis=1) / mother_size
    centroid_pairs=daughter_centroids[daughter_pairs]
    pair_distance = np.abs(1 - np.linalg.norm(centroid_pairs[:,1]-centroid_pairs[:,0], axis=1).flatten() / mother_size)
    pair_distance_diff = np.abs(np.diff(daughter_distances[daughter_pairs]).flatten() / mother_size)
    mother_circs = np.array([mother_circ] * len(daughter_pairs))

    daughter1, daughter2 = daughters[daughter_pairs].T
    return pd.DataFrame(
        {
            'mother': [mother] * len(daughter_pairs),
            'frame': [mother_cell[-1].frame + 1] * len(daughter_pairs),
            'daughter1': daughter1,
            'daughter2': daughter2,
            'mother_circ': mother_circs,
            'daughter_circ': pair_circ,
            'daughter_distance': pair_distance,
            'distance_diff': pair_distance_diff,
            'angle': pair_angle,
            'CoM_displacement': pair_CoM_displacement,
        }
    )


def resolve_conflicts(candidates_df):
    """Resolve conflicts in mitosis candidates"""
    import networkx as nx

    mother_conflicts = candidates_df.groupby('mother').size() > 1
    mother_conflicts = mother_conflicts[mother_conflicts].index
    daughter_conflicts = pd.concat([candidates_df['daughter1'], candidates_df['daughter2']]).value_counts() > 1
    daughter_conflicts = daughter_conflicts[daughter_conflicts].index

    mitosis_graph = nx.Graph()
    for idx, row in candidates_df.iterrows():
        mitosis_graph.add_node(idx, score=row['score'])

    all_conflicts = set()
    for mother in mother_conflicts:
        conflicts = candidates_df[candidates_df['mother'] == mother].index
        all_conflicts.update(combinations(conflicts, 2))

    for daughter in daughter_conflicts:
        conflicts = candidates_df[(candidates_df['daughter1'] == daughter) | (candidates_df['daughter2'] == daughter)].index
        all_conflicts.update(combinations(conflicts, 2))

    mitosis_graph.add_edges_from(all_conflicts)

    best_subset = min_weighted_independent_set(mitosis_graph, weight='score')[0]
    return candidates_df.loc[best_subset]
