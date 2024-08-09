# FlowForge

## Table of Contents
1. [Installation Instructions](#installation-instructions)
2. [Developer Tools](#developer-tools)
3. [License](#license)

## Installation Instructions
Installing flowforge using the following instructions will install the package along with the required dependencies.

### End User Installation
> This will change to PyPI installation once flowforge is open source.
```bash
python -m pip install .
```
### Developer Installation
```bash
python3 -m pip install -e .[dev]
```
### Testing Installation (Primarily for CI)
```bash
python -m pip install .[test]
```

## Developer Tools
The configuration settings for the developer tools can be found in `~/FlowForge/pyproject.toml`.

### Linting Python code with pylint
Execute this line from the `~/FlowForge` directory to lint the code with [pylint](https://pypi.org/project/pylint/):
```bash
pylint ./flowforge
```

### Formatting code with black
Execute this line from the `~/FlowForge` directory to automatically format the code to PEP8 standard using [black](https://pypi.org/project/black/):
```bash
black ./flowforge
```

# License #
[BSD 3-Clause](https://github.com/ut-Computational-NE/FlowForge/blob/master/LICENSE)
