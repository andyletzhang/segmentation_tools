import numpy as np
import networkx as nx

def adjacency_matrix(labeled_array):
    from skimage.segmentation import find_boundaries

    # Get the number of unique labels
    n_labels=labeled_array.max()

    # Initialize adjacency matrix
    adjacency_matrix = np.zeros((n_labels, n_labels), dtype=bool)
    
    # Optionally, find the boundaries
    boundaries = find_boundaries(labeled_array, mode='thick').astype(int)
    boundaries[boundaries > 0] = labeled_array[boundaries > 0]
    # Find adjacency using rolling window (4-connectivity or 8-connectivity)
    for (x, y), label_value in np.ndenumerate(boundaries):
        if label_value > 0: # Exclude background
            neighbors = labeled_array[max(x-1, 0):x+2, max(y-1, 0):y+2].flatten() # 8-connectivity
            for neighbor in neighbors:
                if neighbor != label_value and neighbor > 0:
                    adjacency_matrix[label_value-1, neighbor-1] = True
                    adjacency_matrix[neighbor-1, label_value-1] = True

    return adjacency_matrix

def greedy_color(masks):
    ''' Generate a list of colors for each cell in the segmentation, such that adjacent cells have different colors. '''
    adj=adjacency_matrix(masks)
    G=nx.from_numpy_array(adj)
    colors=nx.coloring.greedy_color(G, strategy='largest_first')

    return [colors[i] for i in np.unique(masks)[1:]-1]

def color_masks(masks, n_colors=20):
    num_masks=masks.max()
    colormap=np.random.randint(n_colors, size=num_masks)

    adj=adjacency_matrix(masks)

    conflict_matrix = (adj & (colormap[:, np.newaxis] == colormap[np.newaxis, :])).astype(int) # adjacent cells with the same color assignments
    conflicts=np.column_stack(np.where(np.triu(conflict_matrix,k=1))) # get the indices of the conflicts

    for i, j in conflicts:
        adjacent_cells=np.where(adj[i])[0]
        adjacent_colors=colormap[adjacent_cells]

        # find the color that is not used by the adjacent cells
        available_colors=np.setdiff1d(np.arange(n_colors), adjacent_colors)

        if len(available_colors)>0:
            colormap[j]=np.random.choice(available_colors)
        else:
            # if all colors are used by adjacent cells, just pick a random color
            colormap[j]=np.random.randint(n_colors)

    return colormap