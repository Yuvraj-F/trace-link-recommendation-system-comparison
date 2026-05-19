# Trace Link Recommendation System Comparison
A package that can evaluate combinations of retireval and ranking techniques for issue-commit trace link recommendation

# Dependencies
`pip install sentence-transformers`. refer to [sbert.net](https://sbert.net/index.html).

# SEOSS 33 Dataset
This package relies on the [SEOSS 33 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/PDDZ4Q).

The simplest method is to place the Seoss33 dataset zip file at the root of the project and it will be automatically detected. If there are 
no zip files or more than one zip file then this detection fails and instead falls back to the `.env`. If the zip file is not at the root of 
the project, then the absolute path to the file can be supplied through the [`.env`](#.env).

# `.env`
Create a `.env` file at the root of the project. The following example shows accepted parameters.
```
COMPRESSED_DATASET_PATH=absolute_path/to/dataset
```