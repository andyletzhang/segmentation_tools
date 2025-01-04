# Onboarding with Python, Part 4: Deep-learning segmentation with Cellpose
*Written by Andy Zhang, last edited December 20, 2024*

This tutorial explains how to install Cellpose, a generalist deep-learning framework for cellular segmentation.

Here I'll just show you how to get Cellpose up and running on your machine. For more guidance on its usage, the [documentation](https://cellpose.readthedocs.io/en/latest/) is quite thorough. And of course feel free to reach out to me and we can segment some stuff together!

## Step 1: Set Up a Dedicated Conda Environment
To avoid dependency conflicts, I highly recommend installing Cellpose in its own Conda environment.

1. **Create a new Conda environment:**
   ```bash
   conda create -n cellpose python=3.10
   ```
   This creates an environment named `cellpose` with Python 3.10, which is what is recommended on the [Cellpose GitHub](https://github.com/mouseland/cellpose). I've used between Python 3.8 and 3.12 without issue, so go with your heart.

2. **Activate the environment:**
   ```bash
   conda activate cellpose
   ```

3. **Install GPU-accelerated PyTorch (Optional)**
   If you have a GPU, you should at this point follow Part 3 of this tutorial, **GPU acceleration with CUDA**, to get PyTorch set up before installing Cellpose itself. Installing PyTorch after installing Cellpose tends to raise issues with dependencies, as Cellpose will default to installing a CPU version of PyTorch.

## Step 2: Install Cellpose
1. **Install Cellpose dependencies (optional)**. We'll install Cellpose using pip since it's not available via `conda install`. However, Cellpose has a bunch of dependencies like numpy, scipy, and numba which it would be best to install using Conda, so that it can manage those installations going forward and make sure we can install other packages on top of Cellpose later on.
   Here's a command that installs stable versions of Cellpose's dependencies manually using Conda (current as of v3.1.0).
   ```bash
   conda install numpy scipy numba tqdm sympy fsspec opencv pyqtgraph qtpy colorama fastremap natsort superqt tifffile
     ```
2. **Install Cellpose**. Cellpose can be installed with or without its graphical frontend, which has a collection of additional dependencies. I think the GUI is pretty solid and worth installing, and we'll use it further on.
   ```bash
   pip install cellpose[gui]
   ```
   If you're just planning on using Cellpose through Python, you can just `pip install cellpose`.

## Step 3: Verify Installation
1. **Check Cellpose version:**
   ```bash
   cellpose --version
   ```
   This command should return the installed version of Cellpose.
2. **Launch GUI (optional):** if you've installed the Cellpose GUI, launch it by running `cellpose` or `python -m cellpose` for older versions. An application should open! If you've set up Cellpose for GPU acceleration, you should get a message which reads `[INFO] ** TORCH CUDA version installed and working. **`