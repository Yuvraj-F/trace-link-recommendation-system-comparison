# Trace Link Recommendation System Comparison
A package that can evaluate combinations of retireval and ranking techniques for issue-commit trace link recommendation

# Database
The simplest method is to place the database zip file at the root of the project and it will be automatically detected. If there are 
no zip files or more than one zip file then this detection fails and instead falls back to the `.env`. 

If the database is not at the root of the project, then the absolute path to the database can be supplied through the `.env`.

# .env
Create a .env file in the root of the project. The following example shows accepted parameters.
```
COMPRESSED_DATABASE_PATH=path/to/databases
```