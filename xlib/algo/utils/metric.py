import skimage.metrics
import numpy as np
import torch
from typing import Union
import torch.nn.functional as F


def calc_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    assert img1.shape == img2.shape, "Image shapes do not match"
    score = skimage.metrics.structural_similarity(img1, img2, channel_axis=2)
    return score


def calc_psnr(
    img1: Union[np.ndarray, torch.Tensor],
    img2: Union[np.ndarray, torch.Tensor],
    max_pixel=1.0,
) -> float:
    if isinstance(img1, torch.Tensor) and isinstance(img2, torch.Tensor):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float("inf")
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()
    elif isinstance(img1, np.ndarray) and isinstance(img2, np.ndarray):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float("inf")
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
