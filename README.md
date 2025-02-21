# Segmentation Tools
_Created by Andy Zhang for use in the [Gardel Lab at the University of Chicago](https://squishycell.uchicago.edu/)._

This repository contains two packages installable via pip: `segmentation_tools`, which provides a suite of analytical tools to segment, track, and analyze fluorescent microscopy data of tissue monolayers, and `segmentation_viewer`, which is a graphical frontend for `segmentation_tools` and offers some additional manual annotation functionality.

![Animated visual of segmented data in the Segmentation Viewer](segmentation_viewer/src/segmentation_viewer/assets/segmentation_viewer_v2.gif)

Segmentation is done using [Cellpose](https://github.com/mouseland/cellpose), and tracking is built on top of [trackpy](https://github.com/soft-matter/trackpy).

## Getting started
### Conda
Before installing the packages, it's recommended to set up a Conda environment. A guide to installing Conda is available [here](python_onboarding/1-anaconda.md).

### GPU acceleration
If you have a GPU, it's also advisable to install a [GPU-accelerated version of PyTorch](python_onboarding/3-GPU.md) to vastly improve the speed of segmentation. You may also want to install [CuPy](https://docs.cupy.dev/en/stable/install.html) which will leverage your GPU to accelerate some computations.

### Installation
`segmentation_viewer` requires the installation of `segmentation_tools`, but `segmentation_tools` can run on its own if you don't plan on using the GUI.

To install `segmentation_tools`, run the following:
```bash
pip install git+https://github.com/andyletzhang/segmentation_tools@main#subdirectory=segmentation_tools
```
To install `segmentation_viewer`:
```bash
pip install git+https://github.com/andyletzhang/segmentation_tools@main#subdirectory=segmentation_viewer
```

## Usage
To launch the GUI, run the command `segmentation-viewer` from within your Conda environment after installation of both packages.

_Last Edited: February 15, 2025_