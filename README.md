# $\mathrm{RISK}_\mathrm{Liver}$-ML
$\mathrm{RISK}_\mathrm{Liver}$ is a ML-based tool to predict hazard levels and potency for drug-induced liver injury(DILI).  

## Structure
- `Descriptors`: packaged for calculating descriptors for input SMILES
- `Viability`: the default working dir, contains models and data to be loaded
- `data_processing.ipynb`: to prepare the input data and calculate descriptors
- `ml_model.ipynb`: make predictions using trained ML models

## Usage
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

# RISKLiver-Server
Backends for prediction on streamlit platform.

## Prerequisites
- You must have an [tortoise-orm supported database](https://tortoise.github.io/databases.html#db-url) running (e.g.: *PostgreSQL*)
    1. Install the database
        ```sh
        sudo docker run -itd -e POSTGRES_PASSWORD=<password> -e POSTGRES_HOST_AUTH_METHOD=trust -v <dir-to-store-data>:/var/lib/postgresql/data -p <port-of-database>:5432 --restart unless-stopped postgres:latest
        ```
    1. Config database  
        Modify `DB` part of `config.json` (detailed format see [tortoise setup](https://tortoise.github.io/setup.html) and [database config instructions](https://tortoise.github.io/databases.html))
- Be sure docker with NVIDIA container toolkit is installed:
    1. Docker: 
        ```sh
        sudo curl -fsSL https://github.com/tech-shrimp/docker_installer/releases/download/latest/linux.sh| bash -s docker --mirror Aliyun
        ```
    1. NVIDIA container toolkit (for Ubuntu):
        ```sh
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt install nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container-tools libnvidia-container1  
        sudo nvidia-ctk runtime configure --runtime=docker  
        sudo systemctl restart docker  
        ```
        Reference: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- Download MATLAB Runtime and EPA OPERA manually:
    - MATLAB Runtime R2022a can be downloaded [here](https://ssd.mathworks.com/supportfiles/downloads/R2022a/Release/9/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2022a_Update_9_glnxa64.zip).
    - EPA OPERA 2.9 CL_Par can be downloaded [here](https://github.com/NIEHS/OPERA/releases/download/v2.9.2/OPERA2.9_CL_Par.tar.xz).
    - After downloading, put the file in the root directory of the repo.  
        The directory structure should be like this:
        ```
        - Descriptors/
        - Estimator/
        - models/
        - StreamlitClient/
        - Viability/
        - config.json
        - Dockerfile
        - env.yml
        - MATLAB_Runtime_R2022a_Update_9_glnxa64.zip
        - OPERA2.9_CL_Par.tar.xz
        - ...
        ```

## Deployment
1. Set up server configuration in `config.json`
1. Build the docker image
    ```sh
    sudo docker build -t riskliver-server .
    ```
1. Run the server  
    In the root path of the repo, run:
    ```sh
    sudo docker run -itd --runtime=nvidia --gpus -p 8003:6310 -v .:/riskliver all --name riskliver-server riskliver-server /root/bin/micromamba run -n riskliver --cwd /riskliver python riskliver_server.py
    ```
    `8003` can be changed to any port you like, just keep the same with the `port` keyword in `config.json` of the fore end.
