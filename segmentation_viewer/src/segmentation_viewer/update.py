from segmentation_tools import __file__ as tools_file
from segmentation_viewer import __file__ as viewer_file
import tempfile
import shutil
from pathlib import Path
import requests
import zipfile
from io import BytesIO
import os

def download_and_extract_repo(repo_url, extract_to):
    # GitHub URL for downloading as a ZIP (change the branch or commit if needed)
    zip_url = f"{repo_url}/archive/refs/heads/main.zip"  # Change `main` to your desired branch

    # Send GET request to download the ZIP file
    response = requests.get(zip_url)
    
    if response.status_code == 200:
        if not os.path.exists(extract_to):
            os.makedirs(extract_to)
        # Use the in-memory ZIP file
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(extract_to)
    return response.status_code

def update_packages():
    # Get package directories
    tools_src=Path(tools_file).parents[1]
    viewer_src=Path(viewer_file).parents[1]

    # Paths to update in repo
    paths_to_include = [
        'segmentation_tools/src/*',
        'segmentation_viewer/src/*'
    ]

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname)

        print(f'Pulling from GitHub...')
        repo = f"andyletzhang/segmentation_tools"  # Replace with your repo
        url = f"https://github.com/{repo}"

        out=download_and_extract_repo(url, tmp_path)
        if out==200:
            print(f'pulled to {tmp_path}')
        else:
            raise ValueError(f"Failed to download repository. Status code: {out}")
        
        # Copy files to appropriate locations
        tools_src_tmp = tmp_path / 'segmentation_tools-main' / 'segmentation_tools' / 'src'
        viewer_src_tmp = tmp_path /  'segmentation_tools-main' / 'segmentation_viewer' / 'src'

        if not tools_src_tmp.exists():
            raise ValueError("segmentation_tools/src not found in repository")
        if not viewer_src_tmp.exists():
            raise ValueError("segmentation_viewer/src not found in repository")
        
        # Copy using shutil.copytree with dirs_exist_ok=True
        # This will merge/overwrite existing directories
        shutil.copytree(tools_src_tmp, tools_src, dirs_exist_ok=True)
        print("Updated segmentation_tools")
        
        shutil.copytree(viewer_src_tmp, viewer_src, dirs_exist_ok=True)
        print("Updated segmentation_viewer")
        
        print("\nPackage update completed successfully!")