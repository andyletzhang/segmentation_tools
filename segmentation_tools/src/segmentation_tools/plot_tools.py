from pathlib import Path

import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from tqdm.notebook import tqdm

from segmentation_tools import preprocessing


def plot_trajectories(
    stack, output_path=None, trail_length=10, linewidth=1, markersize=1, figsize=10, show_labels=True, tracking_kwargs={}
):
    """display trajectories superimposed over images."""
    from matplotlib import collections
    from matplotlib.patches import Polygon

    if not hasattr(stack, 'tracked_centroids'):
        stack.track_centroids(**tracking_kwargs)
    trajectories = stack.tracked_centroids

    if not output_path:
        output_path = stack.name.replace('segmented', 'tracking')
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for n, frame in enumerate(tqdm(stack.frames)):
        frame_drift = stack.drift.loc[n]
        aspect = np.divide(*frame.img.shape[:2])
        plt.figure(figsize=(figsize, figsize * aspect))
        ax = plt.axes()

        if len(frame.img.shape) == 2:
            plt.imshow(preprocessing.normalize(frame.img), cmap='gray')
        elif len(frame.img.shape) == 3:
            if frame.img[..., 2].max() > 0:
                plt.imshow(preprocessing.normalize(frame.img[..., 2]), cmap='gray')  # FUCCI data, blue is membrane channel
            else:
                plt.imshow(preprocessing.normalize(frame.img))  # red nuclei, green membrane
        else:
            raise ValueError(
                f'frame.img has an unexpected number of dimensions {frame.img.ndim}. Can take 2 (grayscale) or 3 (color) dimensions.'
            )

        trails = []  # particle trajectories as plotted on each frame
        for particle, trajectory in trajectories[trajectories.frame <= n].groupby(
            'particle'
        ):  # get tracking data up to the current frame for one particle at a time
            trails.append(
                Polygon(
                    trajectory[['x', 'y']].iloc[-trail_length:] + frame_drift,
                    closed=False,
                    linewidth=linewidth * figsize / 5,
                    fc='none',
                    ec='C{}'.format(particle),
                )
            )

            if show_labels:
                in_frame = trajectory.index[trajectory['frame'] == n]  # get the cell number of the particle in this frame
                try:  # try labeling this particle in this frame
                    cell_number = in_frame[0]  # cell number
                    # TODO: I think the bottleneck is indexing and plotting every text label by hand: can I speed this up somehow? (or I just don't do this too often)
                    particle_label = plt.text(
                        trajectory['x'].loc[cell_number] + 5 + frame_drift['x'],
                        trajectory['y'].loc[cell_number] - 5 + frame_drift['x'],
                        particle,
                        color='white',
                        fontsize=figsize / 2,
                    )
                    particle_label.set_path_effects(
                        [patheffects.Stroke(linewidth=figsize / 5, foreground='C{}'.format(particle)), patheffects.Normal()]
                    )

                except IndexError:  # indexerror means the particle's not identifiable in this frame, move on
                    continue

        # draw trails
        ax.add_artist(collections.PatchCollection(trails, match_original=True))

        # add centroids
        plt.scatter(frame.centroids()[:, 1], frame.centroids()[:, 0], s=np.sqrt(figsize / 5) * markersize, c='white')

        # cosmetics
        xlim = np.array([0, frame.resolution[1]]) + frame_drift['x']
        ylim = np.array([frame.resolution[0], 0]) + frame_drift['y']
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.axis('off')

        plt.savefig(output_path / f'{output_path.stem}-{n}.tif', dpi=300)



def FUCCI_overlay(frame, imshow=True, ax=None, show_denoised=True, show_labels=False, alpha=0.2, normalize=True):
    if not ax:
        ax = plt.gca()
    if imshow:
        if show_denoised:
            color_FUCCI = np.stack([frame.FUCCI[0], frame.FUCCI[1], np.zeros(frame.masks.shape, dtype=int)], axis=-1)
        else:
            if not hasattr(frame, 'img'):
                frame.load_img()
            if normalize:
                norm = preprocessing.normalize
            elif isinstance(normalize, bool):

                def norm(x):
                    return x
            else:
                norm = normalize
            color_FUCCI = np.stack(
                [norm(frame.img[..., 0]), norm(frame.img[..., 1]), np.zeros(frame.masks.shape, dtype=int)], axis=-1
            )
        plt.imshow(color_FUCCI)
    else:
        plt.xlim(0, frame.masks.shape[1])
        plt.ylim(frame.masks.shape[0], 0)

    if show_labels:
        frame.get_centroids()

    cell_overlays = []
    colors = {0: 'none', 1: 'lime', 2: 'r', 3: 'orange'}
    for cell in frame.cells:
        cell_overlays.append(Polygon(cell.outline, ec='white', fc=colors[cell.cycle_stage], linewidth=1, alpha=alpha))

        if show_labels:
            particle_label = plt.text(cell.centroid[1], cell.centroid[0], s=cell.n, color='white', fontsize=6)
            particle_label.set_path_effects(
                [patheffects.Stroke(linewidth=1, foreground=colors[cell.cycle_stage]), patheffects.Normal()]
            )

    ax.add_artist(PatchCollection(cell_overlays, match_original=True))
    plt.axis('off')


# Volume Plots
def volume_boxplot(volumes, labels=None, ax=None, SC_color='C0', ME_color='C2', default_color='C1', **boxplot_kwargs):
    """
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
    """

    if not ax:
        ax = plt.gca()
    if labels is None:
        labels = ['' for _ in volumes]

    default_kwargs = {'showfliers': False, 'patch_artist': True, 'notch': True, 'vert': True}
    boxplot_kwargs = default_kwargs | boxplot_kwargs  # Merge default and user kwargs

    if isinstance(volumes, dict):
        volumes = [np.concatenate(volumes[label]) for label in labels]
    else:
        volumes = [np.concatenate(vol) for vol in volumes]

    bp = ax.boxplot(volumes, labels=labels, **boxplot_kwargs)

    for patch, label in zip(bp['boxes'], labels):
        if label.startswith('SC'):
            color = SC_color
        elif label.startswith('ME'):
            color = ME_color
        else:
            color = default_color
        patch.set_facecolor(color)

    ax.set_xlabel('Volume (μm$^3$)')


def cell_cycle_boxplot(
    volumes, labels=None, ax=None, colors=None, hide_NS=True, ctrl_line=False, xticks='n', trial_labels=True, **boxplot_kwargs
):
    """
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
    """

    if not ax:
        ax = plt.gca()
    if labels is None:
        labels = ['' for _ in volumes]

    if isinstance(volumes, dict):
        volumes = [volumes[label] for label in labels]

    default_kwargs = {'showfliers': False, 'notch': True, 'vert': True}
    boxplot_kwargs = default_kwargs | boxplot_kwargs

    all_bps = []
    for i, vol, label in zip(range(len(labels)), volumes, labels):
        if hide_NS:
            vol = vol[-3:]
        has_NS = not hide_NS and len(vol) == 4
        if colors is None:
            colors = ['g', 'r', 'orange']
            if has_NS:
                colors = ['gray'] + colors

        if has_NS:
            spacing = 5
        else:
            spacing = 4
        positions = np.arange(i * (spacing), (i + 1) * (spacing) - 1)

        if xticks == 'n':
            box_labels = [f'n={len(v)}' for v in vol]
        elif xticks == 'cycle':
            box_labels = ['G1', 'S', 'G2']
            if has_NS:
                box_labels = ['NS'] + box_labels
        else:
            box_labels = ['' for _ in range(len(vol))]

        vol = [v[~np.isnan(v)] for v in vol]
        bp = ax.boxplot(vol, positions=positions, patch_artist=True, labels=box_labels, **boxplot_kwargs)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        all_bps.append(bp)

    if trial_labels:
        ylim = ax.get_ylim()
        y_increment = 0.1 * (ylim[1] - ylim[0])
        ax.set_ylim(ylim[0], ylim[1] + y_increment)
        if trial_labels is True:
            trial_labels = labels
        for i, label in enumerate(trial_labels):
            ax.text(i * spacing + spacing / 2 - 1, ylim[1], label, ha='center', va='center', weight='bold')

    if ctrl_line:
        if ctrl_line is True:
            ctrl_line = 'WT'
        ctrl_vols = [volumes[i] for i in np.where([label.startswith(ctrl_line) for label in labels])[0]]
        if len(ctrl_vols) == 0:
            print('No WT labels found in the list')
        else:
            median_values = [np.nanmedian(np.concatenate([vol[i] for vol in ctrl_vols])) for i in range(3)]
            for med_value, color in zip(median_values, colors):
                ax.axhline(med_value, color=color, linestyle='--', zorder=0)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=50)
    ax.set_ylabel('Volume (μm$^3$)')

    return all_bps


def cell_cycle_occupancy_barplot(
    volumes, labels=None, hide_NS=True, ax=None, xticks='n', colors=None, edge_color='k', **barplot_kwargs
):
    """
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
    """
    if not ax:
        ax = plt.gca()

    if labels is None:
        labels = ['' for _ in volumes]

    if isinstance(volumes, dict):
        volumes = [volumes[label] for label in labels]

    if hide_NS:
        volumes = [vol[-3:] for vol in volumes]
    if isinstance(volumes, dict):
        volumes = [volumes[label] for label in labels]

    occupancies = [[len(v) for v in vol] for vol in volumes]
    total_occupancies = [np.sum(o) for o in occupancies]
    percent_occupancies = np.concatenate([np.array(o) / np.sum(o) for o in occupancies])

    has_NS = not hide_NS and len(volumes[0]) == 4
    condition_spacing = 5 if has_NS else 4

    if colors is None:
        colors = ['g', 'r', 'orange']
        if has_NS:
            colors = ['gray'] + colors

    positions = np.concatenate([np.arange(i * condition_spacing, (i + 1) * condition_spacing - 1) for i in range(len(labels))])

    bars = ax.bar(positions, percent_occupancies, **barplot_kwargs)

    for i in range(condition_spacing - 1):
        for bar in bars[i :: condition_spacing - 1]:
            bar.set_color(colors[i])

    for bar in bars:
        bar.set_edgecolor(edge_color)
    if xticks == 'n':
        ax.set_xticks((np.arange(len(labels)) + 1 / 2) * (condition_spacing) - 1, [f'n={o}' for o in total_occupancies])
    elif xticks == 'cycle':
        cc_labels = ['G1', 'S', 'G2']
        if has_NS:
            cc_labels = ['NS'] + cc_labels
        ax.set_xticks(positions, cc_labels * len(labels))
    return bars


def cell_cycle_plot(
    volumes,
    labels=None,
    axes=None,
    figsize=None,
    hide_NS=True,
    gridspec_kw={'height_ratios': [6, 1]},
    sharex=True,
    boxplot_kwargs={},
    barplot_kwargs={},
    subplot_kwargs={},
    return_patches=False,
):
    """
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
    """
    if not axes:
        if figsize is None:
            figsize = (2 * len(volumes) + 1, 6)
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=sharex, gridspec_kw=gridspec_kw, **subplot_kwargs)

    bps = cell_cycle_boxplot(volumes, labels, ax=axes[0], hide_NS=hide_NS, **boxplot_kwargs)
    bars = cell_cycle_occupancy_barplot(volumes, labels, ax=axes[1], hide_NS=hide_NS, **barplot_kwargs)

    fig.tight_layout()

    if return_patches:
        return fig, axes, bps, bars
    else:
        return fig, axes


def hist_median(
    v,
    histtype='step',
    weights='default',
    zorder=None,
    ax=None,
    bins=30,
    range=(0, 6000),
    median=True,
    linewidth=1.4,
    alpha=1,
    **kwargs,
):
    if not ax:
        ax = plt.gca()
    if isinstance(weights, str) and weights == 'default':
        hist_weights = np.ones_like(v) / len(v)
    else:
        hist_weights = weights
    n, bins, patch = ax.hist(
        v,
        bins=bins,
        range=range,
        zorder=zorder,
        histtype=histtype,
        weights=hist_weights,
        linewidth=linewidth,
        alpha=alpha,
        **kwargs,
    )

    if median:
        med = np.nanmedian(v)
        bin_idx = np.digitize(med, bins) - 1  # find which bin the median is in
        ax.plot(
            [med, med],
            [0, n[bin_idx]],
            color=patch[0].get_edgecolor(),
            zorder=zorder,
            linestyle='--',
            linewidth=linewidth,
            alpha=alpha,
        )  # draw a line at the median up to the height of the bin


def probability_density(x, bins=100, range=None):
    from scipy.stats import gaussian_kde

    x = x[~np.isnan(x)]
    kde = gaussian_kde(x)
    if range is None:
        range = (x.min(), x.max())
    x = np.linspace(*range, bins)
    return kde(x)


def pdf_median(v, ax=None, bins=30, range=(0, 6000), linewidth=1.4, alpha=1, fill_under=False, median=True, **kwargs):
    if not ax:
        ax = plt.gca()
    x = np.linspace(*range, bins)
    pdf = probability_density(v, bins=bins, range=range)
    (line,) = ax.plot(x, pdf, linewidth=linewidth, alpha=alpha, **kwargs)
    if fill_under:
        ax.fill_between(x, pdf, alpha=alpha / 3, color=line.get_color())

    if median:
        med = np.nanmedian(v)
        median_height = pdf[np.argmin(np.abs(x - med))]
        ax.plot([med, med], [0, median_height], color=line.get_color(), linestyle='--')

    return x, pdf
