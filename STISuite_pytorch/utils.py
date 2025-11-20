import os
import torch
from torch import fft
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from typing import List, Tuple, Union, Optional

# STISuite utils for pytorch 1.10
# Jie Feng 21.11.20

Tensor = torch.Tensor
Array = npt.NDArray


@torch.jit.script
def fftnc(x: Tensor, dim: Optional[List[int]] = None) -> Tensor:
    """
    N-dim centered FFT

    :param x: input N-dim Tensor (CPU/GPU)
    :param dim: run FFT in given dim
    :return: output N-dim Tensor (CPU/GPU)
    """
    device = x.device
    if dim is None:
        # this weird code is necessary for torch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i
    return fft.ifftshift(fft.fftn(fft.fftshift(x, dim=dim), dim=dim, norm="ortho"), dim=dim)


@torch.jit.script
def ifftnc(x: Tensor, dim: Optional[List[int]] = None) -> Tensor:
    """
    N-dim centered iFFT

    :param x: input N-dim Tensor (CPU/GPU)
    :param dim: run iFFT in given dim
    :return: output N-dim Tensor (CPU/GPU)
    """
    device = x.device
    if dim is None:
        # this weird code is necessary for torch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i
    return fft.fftshift(fft.ifftn(fft.ifftshift(x, dim=dim), dim=dim, norm="ortho"), dim=dim)


def save_nii(arr: Union[Array, Tensor], filename: str,
             voxel_size: Optional[Union[List[int], Tuple[int]]] = (1, 1, 1),
             affine_3D: Optional[Union[None, Array[float], Tensor, List[float], Tuple[float]]] = None,
             origin: Optional[Union[None, List[float], Tuple[float]]] = None) -> None:
    """
    SimpleITK-based nifti saving. Remember SimpleITK treat the array axis differently.

    Jie Feng

    :param arr: 3D or 4D array of image in (x, y, z, echos). The image should be in LPS space.
    :param filename: saving path and filename
    :param voxel_size: actual voxel resolution of image
    :param affine_3D: affine matrix of image. Should only contains 9 elements.
                      Default = None, means to use default direction of SimpleITK
                      You can get one from Load_QSM, load_nii or sitk.Image.GetDirection()
    :param origin: origin coordinate of image. Should only contains 3 elements.
                      Default = None, means to use default origin of SimpleITK
                      You can get one from Load_QSM, load_nii or sitk.Image.GetOrigin()
    """
    if arr.ndim == 3:
        if isinstance(arr, Tensor):
            img = sitk.GetImageFromArray(arr.permute(2, 1, 0).cpu().numpy())
        elif isinstance(arr, np.ndarray):
            img = sitk.GetImageFromArray(arr.transpose((2, 1, 0)))
        else:
            raise ValueError('the input of save_nii should be np.ndarray or torch.Tensor')
    elif arr.ndim == 4:
        if isinstance(arr, Tensor):
            img = sitk.GetImageFromArray(arr.permute(2, 1, 0, 3).cpu().numpy())
        elif isinstance(arr, np.ndarray):
            img = sitk.GetImageFromArray(arr.transpose((2, 1, 0, 3)))
        else:
            raise ValueError('the input of save_nii should be np.ndarray or torch.Tensor')
    else:
        raise ValueError('save_nii only support 3D or 4D image')

    img.SetSpacing(voxel_size)

    if affine_3D is not None:
        if isinstance(arr, Tensor):
            img.SetDirection(tuple(affine_3D.cpu().numpy().reshape(9,)))
        elif isinstance(arr, np.ndarray):
            img.SetDirection(tuple(affine_3D.reshape(9,)))
        else:
            try:
                img.SetDirection(tuple(np.array(affine_3D).reshape(9,)))
            except:
                raise ValueError('affine_3D of save_nii error')

    if origin is not None:
        img.SetOrigin(origin)
    sitk.WriteImage(img, filename)


def load_nii(filename: str, device: Optional[Union[None, torch.device]] = None,
             numpy_enable: Optional[bool] = False) -> Tuple[Union[Array, Tensor], Union[Array, Tensor], tuple]:
    """
    SimpleITK-based nifti loading. Remember SimpleITK treat the array axis differently.

    Jie Feng

    :param filename: filename of nifti to load. The image will be transformed to LPS space when loading.
    :param device: device of tensor. Only available when numpy_enable is set to False.
                   Default = None, means to use GPU when available
    :param numpy_enable: use numpy or not.
                         When numpy_enable is set to True, the gpu_enable will be invalid
    :return arr: 3D or 4D array/tensor of image in (x, y, z, echos)
            affine_3D: affine matrix of image.
            origin: origin coordinate of image.
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img = sitk.ReadImage(filename)
    img = sitk.DICOMOrient(img, 'LPS')
    arr = sitk.GetArrayFromImage(img)
    affine_3D = np.array(img.GetDirection()).reshape((3, 3))
    origin = img.GetOrigin()
    if numpy_enable:
        if arr.ndim == 3:
            arr = arr.transpose((2, 1, 0))
        elif arr.ndim == 4:
            arr = arr.transpose((2, 1, 0, 3))
        else:
            raise ValueError('load_nii input dim error')
    else:
        if arr.ndim == 3:
            arr = torch.as_tensor(arr).permute(2, 1, 0).to(device)
        elif arr.ndim == 4:
            arr = torch.as_tensor(arr).permute(2, 1, 0, 3).to(device)
        else:
            raise ValueError('load_nii input dim error')
        affine_3D = torch.as_tensor(affine_3D).to(device)
    return arr, affine_3D, origin
