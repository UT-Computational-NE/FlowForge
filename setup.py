from setuptools import setup, find_packages

setup(
    name             = 'flowforge',
    version          = '0.0.1',
    author           = 'Cole Gentry',
    author_email     = 'cole.gentry@austin.utexas.edu',
    description      = 'Frontend interface for multichannel thermal fluids systems solvers',
    url              = 'https://github.com/UT-Computational-NE/FlowForge',
    project_urls     = {
        "Bug Tracker" :  "https://github.com/UT-Computational-NE/FlowForge/issues",
        "Source Code" :  "https://github.com/UT-Computational-NE/FlowForge",
    },
    packages         = find_packages(),
    classifiers      = [
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires  = '>=3.9'
    install_requires = [
        'h5py',
        'numpy>=1.8.0',
        'pyevtk',
    ],
)
