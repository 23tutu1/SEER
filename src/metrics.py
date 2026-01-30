from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse + 1e-8)


def _gaussian_kernel(window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = torch.outer(g, g)
    return kernel


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    if img1.dim() != 4 or img2.dim() != 4:
        raise ValueError("SSIM expects tensors with shape (B, C, H, W).")
    if img1.shape != img2.shape:
        raise ValueError("SSIM expects img1 and img2 to have the same shape.")

    device = img1.device
    window = _gaussian_kernel(window_size, sigma, device)
    window = window.view(1, 1, window_size, window_size)
    window = window.repeat(img1.shape[1], 1, 1, 1)
    padding = window_size // 2

    mu1 = F.conv2d(img1, window, padding=padding, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=padding, groups=img2.shape[1])

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=img1.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=img1.shape[1]) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean(dim=(1, 2, 3))
