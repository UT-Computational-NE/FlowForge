from setuptools import setup, find_packages

# Function to read the contents of the requirements file
def read_requirements():
    with open('requirements.txt') as req:
        content = req.read()
        requirements = [i.strip() for i in content.split('\n') if i.strip() and not i.startswith('#')] # Filter empty lines and comments
    print(requirements)
    return requirements

# Call the function to get the list of requirements
requirements = read_requirements()

setup(
    name             = 'flowforge',
    version          = '0.0.1',
    author           = 'Benjamin Collins',
    author_email     = 'ben.collins@utexas.edu',
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
    python_requires  = '>=3.9',
    install_requires = requirements,
)
