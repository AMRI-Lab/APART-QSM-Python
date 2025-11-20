# STISuite-pytorch
The original STI Suite software package was developed by:

Hongjiang Wei, PhD, Wei Li, PhD, and Chunlei Liu, PhD

---

The python reimplementation was developed by:

Steven Cao, Jie Feng, Guoyan Lao

---
Transformation from numpy to pytorch by:

Jie Feng

**_This version is only available for internal use._**

## Advantages
1. Rewrite the QSM loading function and make an acceleration up to 10x
2. Enable GPU usage and make an acceleration up to 20x in Laplacian-based phase unwrap and V-SHARP
3. Transform all the images to LPS space before calculation and prevent the image from weird flip
4. Add nifti IO function for correct ITK-SNAP visualization
5. Add a unit change in V-SHARP for resolution near 0.1 mm
6. Use typing to make complete annotation (I have made annotation more than ever in this code. -- Jie Feng)

## Requirement

The dependencies of development:
- python 3.8.10 (higher than 3.8 should be fine, or you may have to install typing-extensions for python 3.6 and 3.7)
- pytorch 1.10.0 (higher than 1.7.0 should be fine)
- numpy 1.21.4 (higher than 1.20 should be fine)
- SimpleITK 2.0.2 (higher than 2.0 should be fine)


## Usage

```python
from qsm_funcs import *
```

For detailed usage, please check the [example.py](example.py)

## Bug report

Please report dependency problems or bugs to Jie Feng and Guoyan Lao.
