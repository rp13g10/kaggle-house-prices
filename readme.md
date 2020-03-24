# Kaggle - House Prices
A simple ML model which aims to predict house prices in Ames, USA. The data used in this project were sourced from [this competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

## Requirements
* Python >=3.7.6
* Pandas >=1.0.3
* Matplotlib >=3.2.1
* Seaborn >=0.10.0
* Scikit-learn >=0.22.2
* Tqdm >=4.43.0
* Xgboost >=1.0.2
* Plotly >=4.5.4
* Jupyterlab >=2.0.1

Please note that in order to view figures generated by Plotly in JupyterLab, you will also need to follow [these](https://plot.ly/python/getting-started/#jupyterlab-support-python-35) instructions.

## Project Outline
There are 4 main files in this project, the details of which are as follows:

* eda.py - This contains all of the functions which are used to load and validate the data. Some legacy code from the data discovery process is left in for reference, but is no longer in use.
* etl.py - This contains the functions which are used to transform the data into a suitable format. This includes filling in missing numerical values and performing one-hot encoding on categoricals.
* modeling.py - This contains the functions which were used to determine the best performing models and tune their input parameters. Output predictions are generated as part of this process.
* Summary of Results.ipynb - The primary script for this project. All functions from other files are imported and coordinated here, and any summary figures used in the corresponding blog post were generated using this notebook.