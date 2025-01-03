from segmentation_tools import __file__ as tools_file
from segmentation_viewer import __file__ as viewer_file
import tempfile
import git
import shutil
from pathlib import Path

def update_package():
    # Get package directories
    tools_src=Path(tools_file).parents[1]
    viewer_src=Path(viewer_file).parents[1]
    token_dir=Path(viewer_file).parents[1].parent/'token.txt'

    with open(token_dir, 'r') as f:
        GITHUB_TOKEN=f.read()
    if not GITHUB_TOKEN:
        raise ValueError("GitHub token not found in environment variables")

    # Paths to update in repo
    paths_to_include = [
        'segmentation_tools/src/*',
        'segmentation_viewer/src/*'
    ]

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = Path(tmpdirname)
        # Clone with sparse checkout
        repo = git.Repo.init(tmp_path)
        origin = repo.create_remote('origin', 
            f"https://{GITHUB_TOKEN}@github.com/andyletzhang/segmentation_tools.git")
        
        # Configure sparse checkout
        config = repo.config_writer()
        config.set_value('core', 'sparseCheckout', 'true')
        
        # Write paths to sparse-checkout file
        sparse_checkout_path = Path(repo.git_dir) / 'info' / 'sparse-checkout'
        sparse_checkout_path.parent.mkdir(exist_ok=True)
        with open(sparse_checkout_path, 'w') as f:
            for path in paths_to_include:
                f.write(f"{path}\n")
        
        print("Pulling from GitHub...")
        # Fetch and checkout
        origin.fetch()
        repo.git.checkout('origin/main')
        
        print(f'pulled to {tmp_path}')
        # Copy files to appropriate locations
        tools_src_tmp = tmp_path / 'segmentation_tools' / 'src'
        viewer_src_tmp = tmp_path / 'segmentation_viewer' / 'src'

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
        
        repo.close()
        print("\nPackage update completed successfully!")