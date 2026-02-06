# DeathScope

Repository for died cell detection with apoptosis and necroptosis distinction. It provides training and inference pipelines for training cell detection model as well as all needed data preparation steps.

## cell_adjustor
Contains cell adjustor script that enables us to check parameters of cell images labeling process.
Here are also all the data preprocessing/managment scripts used to prepare the dataset.

### Setup
Python version tested: 3.9.*
Create environment with:    
`python3 -m venv <venv_name>`    
or this for specific version of python installed:    
`python3.9 -m venv <venv_name>`   
Activate the environment with:    
`source <venv_name>/bin/activate`    
Install requirements with:    
`pip install -r requirements_cell_mac.txt`   
or     
`pip install -r requirements_cell_ubuntu.txt`     

## yolo
Yolo training pipeline with custom model wrapper and inference example.
### Setup
Python version tested: 3.8.* for Mac and 3.9.* for Ubuntu.   
#### Mac
Create environment with:    
`python3 -m venv <venv_name>`    
or this for specific version of python installed:    
`python3.8 -m venv <venv_name>`   
Activate the environment with:    
`source <venv_name>/bin/activate`   
Install requirements with:    
`pip install -r requirements_ultralytics_mac.txt`
#### Ubuntu with CUDA
Setup was tested on an Ubuntu 20.04 LTS machine with NVIDIA RTX A5000 and CUDA 12.3.   
Steps for your configuration may differ. Our training/inference setup requires ultralytics package with PyTorch with GPU support.    
Create environment with:    
`python3 -m venv <venv_name>`    
or this for specific version of python installed:    
`python3.9 -m venv <venv_name>`    
Activate the environment with:     
`source <venv_name>/bin/activate`   
Install requirements with:   
`pip3 install torch==2.3.0.dev20240213+cu121 --index-url https://download.pytorch.org/whl/nightly/cu121`    
`pip3 install torchvision==0.18.0.dev20240213+cu121 --index-url https://download.pytorch.org/whl/nightly/cu121`         
`pip3 install ultralytics`           

### Example output
![example1](docs/cell_example.png)
