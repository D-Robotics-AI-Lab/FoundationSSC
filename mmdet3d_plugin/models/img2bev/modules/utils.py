import torch
import numpy as np
import torch.nn as nn


class Voxelization(nn.Module):
    def __init__(self, point_cloud_range, spatial_shape):
        super().__init__()
        self.spatial_shape = spatial_shape
        self.coors_range_xyz = np.array(
            [
                [point_cloud_range[0], point_cloud_range[3]],
                [point_cloud_range[1], point_cloud_range[4]],
                [point_cloud_range[2], point_cloud_range[5]],
            ]
        )

    @staticmethod
    def sparse_quantize(pc, coors_range, spatial_shape):
        idx = spatial_shape * (pc - coors_range[0]) / (coors_range[1] - coors_range[0])
        return idx.long()

    def filter_pc(self, pc, batch_idx):
        def mask_op(data, x_min, x_max):
            mask = (data > x_min) & (data < x_max)
            return mask

        mask_x = mask_op(
            pc[:, 0],
            self.coors_range_xyz[0][0] + 0.0001,
            self.coors_range_xyz[0][1] - 0.0001,
        )
        mask_y = mask_op(
            pc[:, 1],
            self.coors_range_xyz[1][0] + 0.0001,
            self.coors_range_xyz[1][1] - 0.0001,
        )
        mask_z = mask_op(
            pc[:, 2],
            self.coors_range_xyz[2][0] + 0.0001,
            self.coors_range_xyz[2][1] - 0.0001,
        )
        mask = mask_x & mask_y & mask_z
        filter_pc = pc[mask]
        fiter_batch_idx = batch_idx[mask]
        if filter_pc.shape[0] < 10:
            filter_pc = torch.ones((10, 3), dtype=pc.dtype).to(pc.device)
            filter_pc = filter_pc * torch.rand_like(filter_pc)
            fiter_batch_idx = torch.zeros(10, dtype=torch.long).to(pc.device)
        return filter_pc, fiter_batch_idx

    def forward(self, pc, batch_idx):
        pc, batch_idx = self.filter_pc(pc, batch_idx)
        xidx = self.sparse_quantize(
            pc[:, 0], self.coors_range_xyz[0], self.spatial_shape[0]
        )
        yidx = self.sparse_quantize(
            pc[:, 1], self.coors_range_xyz[1], self.spatial_shape[1]
        )
        zidx = self.sparse_quantize(
            pc[:, 2], self.coors_range_xyz[2], self.spatial_shape[2]
        )

        bxyz_indx = torch.stack([batch_idx, xidx, yidx, zidx], dim=-1).long()
        unq, unq_inv, _ = torch.unique(
            bxyz_indx, return_inverse=True, return_counts=True, dim=0
        )

        return unq, unq_inv


def disp2depth_map(disp, baseline, intrins):
    focal = intrins[:, 0, 0]  # [B]
    bf = baseline * focal  # [B]

    depth = bf.view(-1, 1, 1, 1) / (disp + 1e-6)  # [B, D, H, W]

    return depth


def normalize_depth_to_255(depth_map):
    assert (
        depth_map.ndim == 4 and depth_map.size(1) == 1
    ), "depth_map must be (B, 1, H, W)"

    B, _, H, W = depth_map.shape
    
    normalized = torch.zeros_like(depth_map, dtype=torch.float32)

    for b in range(B):
        depth = depth_map[b, 0]  # (H, W)
        valid_mask = depth > 0

        if valid_mask.sum() == 0:
            raise ValueError(f"Batch {b}: No valid depth values found.")

        valid_depth = depth[valid_mask]
        d_min = valid_depth.min()
        d_max = valid_depth.max()

        if d_max == d_min:
            normalized[b, 0][valid_mask] = 255.0
        else:
            normed = (depth[valid_mask] - d_min) / (d_max - d_min) * 255.0
            normalized[b, 0][valid_mask] = normed

    return normalized


def normalize(img, mean, std):
    assert img.ndim == 4, "img must be (B, C, H, W)"

    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=img.device, dtype=img.dtype)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, device=img.device, dtype=img.dtype)

    # (B,C,H,W) -> (B,C,H,W)
    img = img / 255.0
    
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    img = (img - mean) / std

    return img
