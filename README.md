# WRDS accounting data
Python notebook to download data from [Wharton Research Data Services](https://wrds-www.wharton.upenn.edu/)

### Installation

Clone the repository with
`https://github.com/AleBitetto/WRDS_accounting.git`

From console navigate the cloned folder and create a new environment with:
```
conda env create -f environment.yml
conda activate wrds
python -m ipykernel install --user --name wrds --display-name "Python (WRDS)"
```
This will create a `wrds` environment and a kernel for Jupyter Notebook called `Python (WRDS)`

