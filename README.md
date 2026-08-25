# Digital Communications - ECE 632, Fall 2026

This repository is provided for the course ECE 632, Digital Communications, offered in Fall 2026 at George Mason University. Through this repository, students will have access to the Jupyter notebooks, Python scripts, and other resources that will be used in the course. The repository complements the class material distributed through Canvas.

The goal for the course is to design and build a working digital communication system that can transmit and receive digital signals through the audio channel of a computer. The system will be implemented in Python, and students will be expected to collaborate to write code that implements various components of the system, including modulation, demodulation, filtering, and synchronization, equalization, and channel tracking.

## Setting up

### Clone this repository

To clone this repository, first navigate to the directory where you want to clone it, and then use the following command in your terminal:

```bash
git clone git@github.com:bepepa/dc_632_26.git
```

This command assumes that you have set up SSH keys for GitHub. If you have not done so, you can use the HTTPS URL instead:

```bash
git clone https://github.com/bepepa/dc_632_26.git
```

This will create a local copy of the repository in a folder named `dc_632_26`.

### Create a virtual environment

Virtual environments are a way to create isolated Python environments for different projects. This is useful because it allows you to manage dependencies for each project separately, avoiding conflicts between packages and ensuring that your project runs with the correct versions of libraries. It also protects your system's Python installation from being modified by project-specific dependencies.

**Note:** You must have a recent version of Python installed on your system. You can check your Python version by running the following command in your terminal:

```bash
python3 --version
```
Any version of Python 3.8 or later should work for this course.

Navigate to the cloned repository:

```bash
cd dc_632_26
```

Then, create a virtual environment named `.venv` using the following command:

```bash
python3 -m venv .venv
```
### Activate the virtual environment

On macOS and Linux, use the following command:

```bash
source .venv/bin/activate
```

On Windows, use the following command:

```bash
.venv\Scripts\activate
``` 

**Note:** You will need to activate the virtual environment every time you start a new terminal session and want to work on this project. You can deactivate the virtual environment by running the command `deactivate`.

When the virtual environment is activated, you should see the name of the virtual environment (in this case, `.venv`) in your terminal prompt. This indicates that you are now working within the virtual environment.

### Install the required dependencies

To install the required dependencies, run the following command in your terminal:

```bash
pip install -e .
```

The `-e` flag stands for "editable" mode, which means that any changes you make to the source code in the repository will be immediately reflected in the installed package without needing to reinstall it.

This will take a moment to install all the required packages listed in the `pyproject.toml` file. Once the installation is complete, you should see a message indicating that the installation was successful.

### Make a folder for your own notebooks

Create a folder named `my_notebooks` in the root of the repository. This is where you will create your own Jupyter notebooks for the course, e.g., for homework assignments or your own experiments. You can do this using the following command:

```bash
mkdir my_notebooks
```

### Start Jupyter Lab and open the test notebook

To start Jupyter Lab, run the following command in your terminal:

```bash
jupyter lab
```

This will open Jupyter Lab in your default web browser. In Jupyter Lab, navigate to the `notebooks` folder and open the notebook named `00_test_setup.ipynb`. This notebook contains tests to ensure that the required dependencies are installed and working correctly, as well as a simple audio experiment to verify that the audio input and output devices are working correctly.

## Repository structure

There are four main folders in this repository:

* `notebooks`: This folder contains Jupyter notebooks that will be used in the course. The notebooks are organized by topic and will be updated throughout the course.
* `my_notebooks`: This folder is where you will create your own Jupyter notebooks for the course, e.g., for homework assignments or your own experiments. The contents of this folder are not tracked by Git, so you can create and modify notebooks here without affecting the repository.
* `src`: This folder contains Python modules that implement various components of the digital communication system. We will be developing these modules throughout the course, and you will be expected to contribute to their development.
* `tests`: This folder contains unit tests for the Python modules in the `src` folder. You will be expected to write unit tests for any new code that you add to the `src` folder.

## Contributing

You will be expected to contribute to the development of the digital communication system throughout the course. This will involve writing code for various components of the system, as well as writing unit tests for your code. You will also be expected to review and provide feedback on the code written by your classmates.

When contributing to the repository, please follow these guidelines:
* Write clear and concise code that is easy to read and understand.
* Write unit tests for any new code that you add to the `src` folder.
* Use descriptive commit messages that explain the changes you have made.
* Follow the existing code style and conventions used in the repository.
* Use branches for new features or bug fixes, and create pull requests to merge your changes into the main branch.

More details to follow...