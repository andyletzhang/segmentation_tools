import subprocess
import sys

def update_package():
    repo_url = "https://github.com/andyletzhang/segmentation_tools@main"
    for subdir in ["segmentation_tools", "segmentation_viewer"]:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", f"git+{repo_url}#subdirectory={subdir}"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Update failed: {e.stderr}")
            return
    print("Update completed. Restart your application.")