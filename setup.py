from setuptools import setup, find_packages

setup(
    name='meshfleet',
    version='0.1.0',
    author='',
    author_email='your.email@example.com',
    description='Official implementation of the MeshFleet dataset',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/username/objaverse_xl_batched_renderer',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'tqdm',
        'wandb',
        'objaverse',
        'argparse', 
        'numpy',
        'torch',
        'h5py',
        'transformers',
        'ultralytics', 
        'objaverse'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)