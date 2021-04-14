# Courses Discussion

Welcome to the Courses Discussion board, where our community can post any questions, comments, or feedback they have about the experience on OpenMined Courses. Feel free to [navigate over to the discussion](https://github.com/OpenMined/courses/discussions) to get started!

If you have technical issues with the website itself, or want to provide feedback or feature requests, please [create an Issue here](https://github.com/OpenMined/openmined/issues).

## Setup the Environment

It is highly recommended to setup a [virtual environment](https://docs.python.org/3/tutorial/venv.html) where all
the dependecies will be installed. To do so, there are many options available:

- venv (included in Standard Python3, so no extra installation required)
- [conda](https://docs.conda.io/en/latest/miniconda.html) (**Recommended**)

### General Installation Steps

If you're using `conda`, setup the environment (and all the required packages) using the `openmined_courses_conda_env.yml`
YAML file available in the repository

```bash
conda env create -f openmined_courses_conda_env.yml
```

Alternatively, install the code dependencies with `pip`. Also, make sure you're using Python 3.6+.

```bash
pip install -r requirements.txt
```

For now please install Syft like this:

```bash
pip install git+https://github.com/OpenMined/PySyft@dev#egg=syft
```

#### Windows Prerequisites

On Windows, you may require a few extra dependencies. If you're using `conda`, then you should run:

```bash
conda install pywin32
```

If you have issues installing the dependencies or getting the code to run, please ask for help on the [discussion board](https://github.com/OpenMined/courses/discussions)
