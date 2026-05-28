# Trace Link Recommendation System Comparison
A package that can evaluate combinations of retireval and ranking techniques for issue-commit trace link recommendation

# How to run
- Refer to [dataset instructions](#seoss-33-dataset).
- Execute `./run.bat` on Windows or `./run.sh` on Linux/MacOS. Alternatively, if you have a custom system-wide python environment that would be difficult to reinstall in the virtual environment refer to [Python Environment](#python-environment-and-dependencies).

# SEOSS 33 Dataset
This package relies on the [SEOSS 33 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PDDZ4Q).

The simplest method is to place the Seoss33 dataset zip file at the root of the project and it will be automatically detected. 

# `.env` (Optional)
If there are 
no zip files or more than one zip file then this detection fails and instead falls back to the `.env`. If the zip file is not at the root of 
the project, then the absolute path to the file can be supplied through the [`.env`](#.env).

Create a `.env` file at the root of the project. The following example shows accepted parameters.
```
COMPRESSED_DATASET_PATH=absolute_path/to/dataset
```
# Python Environment and Dependencies
The purpose of the batch/bash script is to isolate Python dependencies by creating a virtual environment and installing the required packages from requirements.txt.

However, this setup is not always ideal if you already have a carefully configured system-wide Python environment. In particular, some dependencies (such as PyTorch with CUDA or ROCm support) can be difficult or time-consuming to reinstall correctly inside a new virtual environment.

For example, in my case, I use an AMD GPU and have a working ROCm-enabled PyTorch installation on Python 3.12. Recreating this setup inside a virtual environment would break GPU support, since the environment would not automatically inherit the system-level ROCm configuration.

If you are in a similar situation, you may prefer to bypass the script and install dependencies directly onto your system Python using:
```
python -m pip install -r requirements.txt
```
This avoids rebuilding complex system-specific packages inside a virtual environment, at the cost of reduced dependency isolation.

# Dependencies
All dependencies are included in `requirements.txt`. You can manually install dependencies to your preffered python installation using:
```
python -m pip install -r requirements.txt
```