# APART-QSM-Python

This is the **Python version** code for  
**“APART-QSM: iterative magnetic susceptibility sources separation”.**

This repository provides a Python + PyTorch implementation of the APART-QSM reconstruction pipeline.

This project contains pre-compiled .pyc files and TorchScript functions. To ensure compatibility, the following versions are strictly required: **Python 3.8**.
All additional dependencies are listed in: requirements_wogpu.txt

Below is an example of the required data layout for running `demo.py`:
```text
data/
├── GRE/ # Multi-echo GRE DICOM directory
│ ├── IM0001.dcm
│ ├── IM0002.dcm
│ ├── IM0003.dcm
│ ├── ...
│ └── IM00NN.dcm
│
└── T2.nii.gz # T2 map used in APART-QSM
