import numpy as np
from interp3d_func import interp3d
from scipy import io
mat = io.loadmat('C:/Users/FengJie/Desktop/test_interp3.mat')
input = mat['Phase']
input_ = np.ascontiguousarray(input)
tmp = interp3d.interp3DtoStd(input_,np.array([0.75,0.75,1.5]))
result = mat['PhaseUWPUpsampled']