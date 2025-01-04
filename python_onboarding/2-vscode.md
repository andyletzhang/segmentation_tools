# Onboarding with Python, Part 2: Visual Studio Code
*Written by Andy Zhang, last edited December 19, 2024*

This section is optional but highly recommended. Once you get comfortable with VSCode, you'll have a lot of useful new functionality at your fingertips such as syntax error correction and code suggestions from GitHub Copilot.

## Installation
To install VSCode:
1. Visit the [Visual Studio Code Download page](https://code.visualstudio.com/Download).
2. Select the installer for your operating system and run it.
3. Follow the on-screen instructions, accepting default settings unless you have specific preferences.

After installation, open VSCode to familiarize yourself with its interface. You may be prompted to install recommended extensions, which is a good starting point.

## Opening a Project Folder
You can open individual files by dragging and dropping or `File -> Open File...` but in VSCode, projects are organized as folders (which should probably square with however you're storing your Python scripts). To begin working on a project, click on `File -> Open Folder...` and select the root directory you want to work in. Once the folder is open, you will see its contents listed in the Explorer panel on the left-hand side. Working within a project folder helps VSCode manage paths and settings nicely. (Rest assured that you can still access files in other directories within your sPython scripts by giving the full file path!)

## Plugins
You'll need the Python and Jupyter plugins to execute iPython notebooks. These are available to install in your left toolbar. The extensions marketplace also offers a bunch of other free, user-provided augmentations to your workflow ranging from visual overhauls to syntax correctors to support for more filetypes... all sorts of stuff. If you're interested, I encourage you look around the marketplace or look up "top Python VSCode extensions" online. I personally have a few small quality-of-life extensions that I like, mostly for the visual front-end: Bookmarks, vscode-icons, Rainbow CSV, Markdown Preview Enhanced, but it depends on the person!

## Connecting your Python Environment to VSCode
Now that you've opened your project folder, open or create a `.ipynb` file. In the notebook interface, locate the kernel selector at the top-right corner of the screen. This allows you to select the Python environment (kernel) for running the notebook. Choose the kernels corresponding to your Conda environment. The kernel names usually include the Conda environment name. For running of `.py` scripts and proper syntax highlighting, you may also have to specify a Python interpreter (which is usually the same virtual environment); press `Ctrl+Shift+P` and type "interpreter." An option should pop up titled `Python: Select Interpreter` which will again allow you to select your virtual environment.

## GitHub Copilot
GitHub Copilot is an AI-powered code assistant that can suggest lines or blocks of code based on the context. To access Copilot:
1. Visit the [GitHub Education Pack](https://education.github.com/pack) to check your eligibility for free access as a student or researcher. It may take you up to two weeks to get approved.
2. Install the Copilot extension in VSCode.
3. Follow the setup instructions to link your GitHub account.
4. Open a notebook or Python file and start typing. If everything's set up properly, you should see a little Copilot icon in the bottom toolbar on the right which displays a loading ring, and then a code suggestion should show up!
5. There's also a chatbot available in the vertical toolbar (usually on the left) which you can talk to and which will use whatever highlighted code you're working on in its response.