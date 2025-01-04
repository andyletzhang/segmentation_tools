# Onboarding with Python, Part 1: Anaconda
*Written by Andy Zhang, last edited December 19, 2024*

## Installing Anaconda
**Anaconda** is a robust platform for managing Python environments and packages. It will install Python and everything else you'll need to get started. To begin:
- Visit the [Anaconda Download page](https://www.anaconda.com/download/success) and choose the installer appropriate for your operating system (Windows, macOS, or Linux).
- Run the installer, accepting default settings unless specific needs arise. 
- After installation, verify success by opening the now-installed **Anaconda Powershell Prompt** and typing:
  ```bash
  conda --version
  ```
- If you see a version number and not an error, you're good to go.

## Anaconda Prompt
Anaconda (Powershell) Prompt* serves as a command-line interface tailored for tasks within Anaconda. Use it to manage environments, install packages, and launch applications such as Jupyter Notebook. On Windows, search for "Anaconda Prompt" in the Start Menu; on macOS or Linux, open a terminal and ensure Anaconda's `bin` directory is in your PATH.

**Anaconda Prompt and Anaconda Powershell Prompt are two separate applications installed by Anaconda on Windows machines. The baseline functionality of the two are equivalent, but Anaconda Prompt is a legacy application built on top of cmd and is less versatile. I recommend Powershell.*

## Virtual Environments
Virtual environments are crucial for isolating projects and managing dependencies. Each virtual environment is its own instance of Python and all its associated packages. Packages often require conflicting versions of Python or libraries, making environments essential for stability. For instance, a package for building neural networks might need Python 3.8, while another for image segmentation requires 3.10. Keeping these in separate environments avoids conflicts.

Environments should be minimal, containing only the necessary dependencies for the task. This reduces the likelihood of compatibility issues and keeps them lightweight.

When installed, Anaconda will by default create a (base) environment for you. It's best practice not to install anything additional into this environment!

To create a virtual environment with name "myenv" (change this to whatever makes sense for you) and have it use version 3.11 of Python*, write:
```bash
conda create --name myenv python=3.11
```
Activate it with:
```bash
conda activate myenv
```
It's here that you'll run scripts and install packages. When finished, deactivate (and return to the base environment) with:
```bash
conda deactivate
```
You can list all available environments using:
```bash
conda env list
```
You can find more tools for deleting, cloning, and exporting environments in the [Anaconda Documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

The currently active environment will be indicated by an environment name in parentheses on the left of the current directory (e.g. `(base) PS C:\Users\Andy\jupyter>`.) Every time Anaconda Prompt is opened it will begin in your base environment, so remember to check where you are before you start running/installing stuff.

**regarding Python versioning: some packages which are more volatile and actively developed might require a specific version of Python, but stable packages like numpy, scipy, and matplotlib will probably not be too sensitive to this. I primarily work in Python 3.11. You may run into instllation issues working with newer (3.12) or older (3.8 or older) versions.*

## Package Management
You have several options available for installing packages (e.g. numpy, matplotlib, cellpose):
- **Conda:** installing packages using `conda install` attempts to solve a network of all the known dependencies; it will determine all the dependencies and their versions so that your packages get along. Use this whenever possible for stability and compatibility.
  ```bash
  conda install numpy
  ```
- **Pip:** `pip install` accesses the Python Package Index (PyPI) and can install basically any package which is publicly available. However, it's less delicate about preserving compatibility. Use pip when a package isn’t available in conda.
  ```bash
  pip install cellpose
  ```
- **Git + Pip:** For custom forks or packages not available from PyPI, you can `pip install` directly from repositories hosted on GitHub or GitLab.
  ```bash
  pip install git+https://github.com/user/repo.git
  ```

## Jupyter Notebook and Jupyter Lab
Jupyter Notebook and Jupyter Lab are powerful tools for interactive programming, and come prepackaged with Anaconda. They allow you to combine code, visualizations, and explanatory text/markdown in a single document (called an **IPython notebook**), which is particularly useful for exploratory data analysis and creating reproducible workflows.

Using an IPython notebook (`.ipynb` file) has several advantages over running standalone Python scripts. Notebooks provide an interactive environment where you can execute code in small increments (cells), which is super helpful for debugging and doing analysis/visualization on the fly. They also allow inline visualizations and Markdown text for clear documentation.

Jupyter Lab/Notebook have the same core functionality. Jupyter Lab offers a more modern interface with advanced features, while Jupyter Notebook is more barebones. If you're choosing between using one of these two, perhaps use Jupyter Lab as it is more actively supported.

To launch Jupyter, go to your base environment where it's installed by default and type:
```bash
jupyter notebook
```
or
```bash
jupyter lab
```

Jupyter is all you need to start doing data science in Python, but if you want more power and customizability, we'll introduce **Visual Studio Code (VSCode)** later on. VSCode is an industry-standard code editor that supports IPython notebooks (along with a ton of other features including integration with GitHub, AI coding assistants, and quality of life improvements like visual customizations, debugging tools, and more powerful find and replace.)

## Using Environments in Jupyter
Jupyter will run your base environment Python by default. To import the packages that you've installed in a virtual environment, you will need to install the **IPython kernel** within that environment. For example, to add the environment "myenv" as an IPython kernel, activate that environment and run the following lines:
```bash
conda activate myenv
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"
```
(`--display-name` is an optional cosmetic description of the kernel; make this whatever makes sense to you.) This adds the environment as a selectable option in the top right corner of your Jupyter window. Restart Jupyter to see the new kernel in the dropdown.