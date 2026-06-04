# RISKLiver-ML
RISKLiver is a ML-based tool to predict hazard levels and potency for drug-induced liver injury(DILI).

# Structure
- `Descriptors`: packaged for calculating descriptors for input SMILES
- `Viability`: the default working dir, contains models and data to be loaded
- `data_processing.ipynb`: to prepare the input data and calculate descriptors
- `ml_model.ipynb`: make predictions using trained ML models

# Usage
1. Create a conda environment using `env.yml`  
    ```bash
    conda env create -f env.yml -n RISKLiver
    ```
    > Community alternative of anaconda may works fine as well  
1. Unzip the models in `Viability/models`
    ```bash
    cd Viability/models
    unzip Viability/models/models.zip
    ```
1. Read the instructions in `data_processing.ipynb` and `ml_model.ipynb`, and run the notebooks
