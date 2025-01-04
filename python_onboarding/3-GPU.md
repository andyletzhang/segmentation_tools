# Onboarding with Python, Part 3: GPU acceleration with CUDA
*Written by Andy Zhang, last edited December 20, 2024*

This tutorial guides you through setting up PyTorch with CUDA to leverage your NVIDIA GPU for segmentation and deep learning. We'll start from right after you've physically installed your GPU on your motherboard (please get help doing this if you haven't done it before!).

## Step 1: Install NVIDIA Drivers
To use CUDA, you need compatible NVIDIA GPU drivers installed on your system:

1. **Check for existing drivers:**
   ```bash
   nvidia-smi
   ```
   If this command returns information about your GPU, the drivers are already installed.

2. **Download and install drivers:**
   - Visit the [NVIDIA Driver Downloads page](https://www.nvidia.com/Download/index.aspx). Here you can choose to either install NVIDIA's driver manager, or just install their latest drivers.
   - Manual driver installation:
     - Select your GPU model and operating system.
     - Download and install the recommended driver.
   - Driver manager:
     - On Linux, install `nvidia-driver-manager` through your package manager. For example, on Ubuntu:
     ```bash
     sudo apt install nvidia-driver-manager
     ```
     - On Windows, use the [NVIDIA App](https://www.nvidia.com/en-us/software/nvidia-app-enterprise/) to manage and update drivers automatically.
3. **Reboot after installation** to apply changes. Run `nvidia-smi` again and verify that your GPU is being detected.

## Step 2: Install PyTorch with CUDA
In theory, you should be able to install a prepackaged version of PyTorch which works with CUDA. Your mileage may vary, but this is a good place to start.

1. **Install PyTorch using Conda:**
   I highly recommend you do this in an empty environment with just python in it: e.g. `conda create -n myenv python=3.11`. If installing PyTorch doesn't work properly, the culprit is almost certainly some lurking dependency from another package.
   In the following, I will use v11.8 of CUDA, which works for everything I need. Replace this at your own risk!
   ```bash
   conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

2. **Verify the installation:**
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```
   Ensure `torch.cuda.is_available()` returns `True`.

## Troubleshooting
If everything has installed okay, but `torch.cuda.is_available()` returns `False`, it may be that conda has failed to install the proper versions of CUDA and PyTorch. In this case I recommend installing the component parts one by one.

1. **Install CUDA Toolkit**
   You may try a CUDA version aside from 11.8 if you want, just make sure the version of cudatoolkit and pytorch-cuda are the same.
   ```bash
   conda install cudatoolkit=11.8 -c nvidia
   ```
2. **Install PyTorch-CUDA Compatibility Package**
   ```bash
   conda install pytorch-cuda=11.8 -c nvidia
   ```
3. **Install PyTorch**
   ```bash
   conda install pytorch -c pytorch
   ```
   In this step, Conda should show some indication that it's recognized it should install a CUDA-enabled version of PyTorch. Look for a message like this pending confirmation, where `cuda11.8` is embedded in the PyTorch version:
   ```
   The following NEW packages will be INSTALLED:
     ...
     pytorch            pytorch/win-64::pytorch-2.5.1-py3.11_cuda11.8_cudnn9_0
     ...
   ```
   After this installs, run `torch.cuda.is_available()` again. Hopefully you're all set!