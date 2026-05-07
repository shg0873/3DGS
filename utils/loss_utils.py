#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils.general_utils import build_scaling_rotation
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

# ============================================================
# L1 Loss
# ============================================================
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

# ============================================================
# Gaussian 커널 생성
# ============================================================
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
# ============================================================
# SSIM 계산 함수
# ============================================================
def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
        
# ============================================================
# Sobel 기반 edge field 생성
# ============================================================
def _sobel_edge_fields(image, percentile=85.0, power=1.0):

    r, g, b = image[0:1], image[1:2], image[2:3]      # RGB Channel
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b       # grayscale 

    # Create Sobel Kernel
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=image.device, dtype=image.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=image.device, dtype=image.dtype).view(1, 1, 3, 3)

    # Gradient 계산
    gx = F.conv2d(gray.unsqueeze(0), kx, padding=1)   # x 방향 gradient
    gy = F.conv2d(gray.unsqueeze(0), ky, padding=1)   # y 방향 gradient
    grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-12).squeeze(0).squeeze(0)  # gradient magnitude 계산

    # --------------------------------------------------------
    # edge 강도 맵 및 edge normal 생성
    # --------------------------------------------------------
    flat = grad_mag.reshape(-1)
    if percentile > 0.0:
        q = torch.quantile(flat, torch.clamp(torch.tensor(percentile / 100.0, device=image.device), 0.0, 1.0))
        score = (grad_mag - q).clamp_min(0.0)
    else:
        score = grad_mag

    denom = score.max().clamp_min(1e-6)
    edge_w = (score / denom).pow(power)

    gx2 = gx.squeeze(0).squeeze(0)
    gy2 = gy.squeeze(0).squeeze(0)
    gn = torch.sqrt(gx2 * gx2 + gy2 * gy2 + 1e-12)
    nx = gx2 / gn
    ny = -gy2 / gn
    return edge_w, nx, ny

# ============================================================
# Edge-aware covariance loss
# ============================================================
def edge_aware_covariance_loss(gt_image, xyz, scaling, rotation, radii, full_proj_transform, world_view_transform, fovx, fovy,
    edge_percentile=85.0, edge_power=1.0, min_weight=0.0,):

    # edge 정보 생성
    edge_w_map, edge_nx_map, edge_ny_map = _sobel_edge_fields(gt_image, percentile=edge_percentile, power=edge_power)
    h, w = edge_w_map.shape

    ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
    xyz_h = torch.cat([xyz, ones], dim=1)

    # clip space projection
    clip = xyz_h @ full_proj_transform
    clip_w = clip[:, 3]
    valid_w = clip_w.abs() > 1e-8

    # 2D 좌표
    ndc_xy = clip[:, :2] / safe_w.unsqueeze(1)
    
    grid = ndc_xy[valid].view(1, -1, 1, 2)
    sampled_w = F.grid_sample(edge_w_map.view(1, 1, h, w), grid, mode="bilinear",
        padding_mode="zeros",align_corners=True).view(-1)
    sampled_nx = F.grid_sample(edge_nx_map.view(1, 1, h, w), grid, mode="bilinear",
        padding_mode="zeros", align_corners=True).view(-1)
    sampled_ny = F.grid_sample(edge_ny_map.view(1, 1, h, w), grid, mode="bilinear",
        padding_mode="zeros", align_corners=True).view(-1)

    if min_weight > 0.0:
        keep = sampled_w >= float(min_weight)
        if keep.sum() == 0:
            return torch.zeros((), device=gt_image.device)
        sampled_w = sampled_w[keep]
        sampled_nx = sampled_nx[keep]
        sampled_ny = sampled_ny[keep]
        valid_indices = keep
    else:
        valid_indices = None

    # camera space 좌표
    view_xyz = (xyz_h @ world_view_transform)[:, :3]
    if valid_indices is not None:
        view_xyz_valid = view_xyz[valid][valid_indices]
        scaling_valid = scaling_vis[valid][valid_indices]
        rotation_valid = rotation_vis[valid][valid_indices]
    else:
        view_xyz_valid = view_xyz[valid]
        scaling_valid = scaling_vis[valid]
        rotation_valid = rotation_vis[valid]

    # z > 0 인 점만 사용
    valid_z = view_xyz_valid[:, 2] > 1e-6
    if valid_z.sum() == 0:
        return torch.zeros((), device=gt_image.device)

    sampled_w = sampled_w[valid_z]
    sampled_nx = sampled_nx[valid_z]
    sampled_ny = sampled_ny[valid_z]
    view_xyz = view_xyz_valid[valid_z]
    scaling_vis = scaling_valid[valid_z]
    rotation_vis = rotation_valid[valid_z]

    # edge normal 벡터 n=(nx, ny)
    n = torch.stack([sampled_nx, sampled_ny], dim=1)
    n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-6)
    # 3D Gaussian의 3차원 공분산 행렬
    L = build_scaling_rotation(scaling_vis, rotation_vis)
    # world covariance Σ_world = L Lᵀ
    cov_world = L @ L.transpose(1, 2)
    # camera covariance 변환 Σ_camera = Rᵀ Σ_world R
    a = world_view_transform[:3, :3]
    cov_cam = a.transpose(0, 1).unsqueeze(0) @ cov_world @ a.unsqueeze(0)

    # tangent 계산
    tan_x = torch.tan(torch.tensor(0.5 * fovx, device=xyz.device, dtype=xyz.dtype)).clamp_min(1e-6)
    tan_y = torch.tan(torch.tensor(0.5 * fovy, device=xyz.device, dtype=xyz.dtype)).clamp_min(1e-6)
    x = view_xyz[:, 0]
    y = view_xyz[:, 1]
    z = view_xyz[:, 2]
    # projection Jacobian
    # u = x / (tan_x * z), v = y / (tan_y * z)
    j = torch.zeros((view_xyz.shape[0], 2, 3), device=xyz.device, dtype=xyz.dtype)
    j[:, 0, 0] = 1.0 / (tan_x * z)    # ∂u/∂x = 1 / (tan_x z)
    j[:, 0, 2] = -x / (tan_x * z * z) # ∂u/∂z = -x / (tan_x z²)
    j[:, 1, 1] = 1.0 / (tan_y * z)    # ∂v/∂y = 1 / (tan_y z)
    j[:, 1, 2] = -y / (tan_y * z * z) # ∂v/∂z = -y / (tan_y z²)
    # 2D covariance Σ₂D
    cov_ndc = j @ cov_cam @ j.transpose(1, 2)  
    # edge normal 방향 분산  σ⊥ = nᵀ Σ_2D n
    sigma_perp = torch.einsum("bi,bij,bj->b", n, cov_ndc, n).clamp_min(0.0) 
    # edge weight 적용
    weighted = sampled_w * sigma_perp
    return weighted.sum() / (sampled_w.sum() + 1e-12)


def sample_edge_weights_for_gaussians(gt_image,xyz,radii,
    full_proj_transform,edge_percentile=85.0,edge_power=1.0):
    # 1. GT 이미지로부터 Sobel 기반 edge weight map 생성
    edge_w_map, _, _ = _sobel_edge_fields(gt_image, percentile=edge_percentile, power=edge_power)
    h, w = edge_w_map.shape
        
    # 2. edge 정보 생성
    edge_weights = torch.zeros((xyz.shape[0],), device=xyz.device, dtype=xyz.dtype)
    xyz_vis = xyz[vis]
    ones = torch.ones((xyz_vis.shape[0], 1), device=xyz_vis.device, dtype=xyz_vis.dtype)
    xyz_h = torch.cat([xyz_vis, ones], dim=1)

    # clip space projection
    clip = xyz_h @ full_proj_transform
    clip_w = clip[:, 3]

    safe_w = torch.where(clip_w >= 0.0, clip_w.clamp_min(1e-8), clip_w.clamp_max(-1e-8))
    ndc_xy = clip[:, :2] / safe_w.unsqueeze(1)
    # 화면 내부에 존재하는 Gaussian만 선택
    in_bounds = (ndc_xy[:, 0].abs() <= 1.0) & (ndc_xy[:, 1].abs() <= 1.0)
    valid = valid_w & in_bounds
    if valid.sum() == 0:
        return edge_weights

    grid = ndc_xy[valid].view(1, -1, 1, 2)
    sampled_w = F.grid_sample(
        edge_w_map.view(1, 1, h, w),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(-1)

    vis_idx = vis.nonzero(as_tuple=False).squeeze(1)
    valid_idx = vis_idx[valid]
    edge_weights[valid_idx] = sampled_w
    return edge_weights


def sobel_edge_weight_map(image, percentile=85.0, power=1.0):
    edge_w, _, _ = _sobel_edge_fields(image, percentile=percentile, power=power)
    return edge_w
