from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RenderResult:
    rgb: torch.Tensor
    depth: torch.Tensor
    acc: torch.Tensor
    rgb_coarse: Optional[torch.Tensor] = None
    depth_coarse: Optional[torch.Tensor] = None
    acc_coarse: Optional[torch.Tensor] = None


def sample_along_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    num_samples: int,
    near: float,
    far: float,
    perturb: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = rays_o.device
    t_vals = torch.linspace(0.0, 1.0, steps=num_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals
    z_vals = z_vals.expand(rays_o.shape[0], num_samples)

    if perturb:
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, :1], mids], dim=-1)
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]
    return pts, z_vals


def sample_pdf(bins: torch.Tensor, weights: torch.Tensor, num_samples: int) -> torch.Tensor:
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=cdf.device)
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)

    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(*inds_g.shape[:-1], cdf.shape[-1]), -1, inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(*inds_g.shape[:-1], bins.shape[-1]), -1, inds_g)
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    return bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])


def volume_render(
    raw: torch.Tensor,
    z_vals: torch.Tensor,
    rays_d: torch.Tensor,
    white_bkgd: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma = F.relu(raw[..., 0])
    rgb = torch.sigmoid(raw[..., 1:4])

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    alpha = 1.0 - torch.exp(-sigma * dists)
    trans = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1,
    )[..., :-1]
    weights = alpha * trans

    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * z_vals, dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, depth_map, acc_map, weights


def _run_radiance_field(
    model: nn.Module,
    pts: torch.Tensor,
    dirs: torch.Tensor,
    clip_embedding: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if clip_embedding is None:
        return model(pts, dirs)
    return model(pts, dirs, clip_embedding)


def render_rays(
    model_coarse: nn.Module,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    num_samples: int,
    near: float,
    far: float,
    perturb: bool = False,
    chunk_rays: Optional[int] = None,
    clip_embedding: Optional[torch.Tensor] = None,
    white_bkgd: bool = False,
    num_importance: int = 0,
    model_fine: Optional[nn.Module] = None,
) -> RenderResult:
    n_rays = rays_o.shape[0]
    chunk = chunk_rays or n_rays
    rgb_chunks = []
    depth_chunks = []
    acc_chunks = []
    rgb_coarse_chunks = []
    depth_coarse_chunks = []
    acc_coarse_chunks = []

    clip_all = clip_embedding
    if clip_all is not None and clip_all.shape[0] == 1 and n_rays > 1:
        clip_all = clip_all.expand(n_rays, -1)

    for i in range(0, n_rays, chunk):
        rays_o_chunk = rays_o[i : i + chunk]
        rays_d_chunk = rays_d[i : i + chunk]
        clip_chunk = clip_all[i : i + chunk] if clip_all is not None else None

        pts, z_vals = sample_along_rays(
            rays_o_chunk, rays_d_chunk, num_samples, near, far, perturb
        )
        dirs = rays_d_chunk[:, None, :].expand_as(pts)
        raw = _run_radiance_field(model_coarse, pts, dirs, clip_chunk)
        rgb_coarse, depth_coarse, acc_coarse, weights = volume_render(
            raw, z_vals, rays_d_chunk, white_bkgd=white_bkgd
        )

        if num_importance > 0:
            z_vals_mid = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
            if z_vals_mid.shape[-1] < 2:
                rgb_chunks.append(rgb_coarse)
                depth_chunks.append(depth_coarse)
                acc_chunks.append(acc_coarse)
            else:
                pdf_weights = weights[..., 1:-1]
                z_samples = sample_pdf(z_vals_mid, pdf_weights, num_importance)
                z_vals_fine, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
                pts_fine = rays_o_chunk[:, None, :] + rays_d_chunk[:, None, :] * z_vals_fine[..., None]
                dirs_fine = rays_d_chunk[:, None, :].expand_as(pts_fine)
                raw_fine = _run_radiance_field(
                    model_fine or model_coarse, pts_fine, dirs_fine, clip_chunk
                )
                rgb_fine, depth_fine, acc_fine, _ = volume_render(
                    raw_fine, z_vals_fine, rays_d_chunk, white_bkgd=white_bkgd
                )
                rgb_chunks.append(rgb_fine)
                depth_chunks.append(depth_fine)
                acc_chunks.append(acc_fine)
        else:
            rgb_chunks.append(rgb_coarse)
            depth_chunks.append(depth_coarse)
            acc_chunks.append(acc_coarse)

        rgb_coarse_chunks.append(rgb_coarse)
        depth_coarse_chunks.append(depth_coarse)
        acc_coarse_chunks.append(acc_coarse)

    rgb_map = torch.cat(rgb_chunks, dim=0)
    depth_map = torch.cat(depth_chunks, dim=0)
    acc_map = torch.cat(acc_chunks, dim=0)
    rgb_coarse = torch.cat(rgb_coarse_chunks, dim=0)
    depth_coarse = torch.cat(depth_coarse_chunks, dim=0)
    acc_coarse = torch.cat(acc_coarse_chunks, dim=0)

    return RenderResult(
        rgb=rgb_map,
        depth=depth_map,
        acc=acc_map,
        rgb_coarse=rgb_coarse,
        depth_coarse=depth_coarse,
        acc_coarse=acc_coarse,
    )


def multiview_consistency_loss(
    nerf: nn.Module,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    num_samples: int,
    near: float,
    far: float,
    color_weight: float = 0.0,
) -> torch.Tensor:
    if rays_o.shape[0] < 2:
        return torch.tensor(0.0, device=rays_o.device)

    pts, _ = sample_along_rays(rays_o, rays_d, num_samples, near, far, perturb=False)
    dirs = rays_d[:, None, :].expand_as(pts)

    perm = torch.randperm(rays_d.shape[0], device=rays_d.device)
    dirs_perm = rays_d[perm][:, None, :].expand_as(pts)

    raw_a = nerf(pts, dirs)
    raw_b = nerf(pts, dirs_perm)
    sigma_a = F.relu(raw_a[..., 0])
    sigma_b = F.relu(raw_b[..., 0])
    loss = F.l1_loss(sigma_a, sigma_b)

    if color_weight > 0.0:
        rgb_a = torch.sigmoid(raw_a[..., 1:4])
        rgb_b = torch.sigmoid(raw_b[..., 1:4])
        loss = loss + color_weight * F.l1_loss(rgb_a, rgb_b)
    return loss
