from setuptools import setup, find_packages

setup(
    name='segmentation_viewer',
    version='0.0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'segmentation-viewer = segmentation_viewer.main:main',
        ],
    },
    install_requires=[
        'PyQt6',
        'PyOpenGL',
        'shapely',
        'numpy',
        'superqt',
        'monolayer_tracking'
    ],
    package_data={
        'segmentation_viewer': ['assets/*'], # Include the assets directory in the package
    },
)