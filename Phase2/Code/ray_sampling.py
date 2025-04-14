import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_rays(H: int, W: int, focal: float, c2w: torch.Tensor):
    """
    Generate camera rays for each pixel in an image.
    """
    c2w = c2w.to(DEVICE)
    
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=DEVICE),
        torch.arange(H, dtype=torch.float32, device=DEVICE),
        indexing='ij'
    )
    
    dirs = torch.stack([
        (i - W / 2) / focal,
        -(j - H / 2) / focal,
        -torch.ones_like(i)
    ], dim=-1)
    
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

def sample_rays(rays_o: torch.Tensor, rays_d: torch.Tensor, target_pixel_values: torch.Tensor, num_rays: int):
    """
    Randomly sample a fixed number of rays from the provided set.
    """
    N = rays_o.shape[0]
    indices = torch.randperm(N, device=DEVICE)[:num_rays]
    return rays_o[indices], rays_d[indices], target_pixel_values[indices]

def generateBatch(images: np.ndarray, poses: np.ndarray, focal: float) -> torch.Tensor:
    """
    Generate a batch of rays from a set of images and corresponding camera poses.
    
    Args:
        images (np.ndarray): (N, H, W, 3) in [-1,1] space.
        poses (np.ndarray): (N, 4, 4).
        focal (float).
    
    Returns:
        all_rays (torch.Tensor): shape (total_pixels, 9) = [ray_o(3), ray_d(3), pixel(3)].
    """
    ray_origins_list = []
    ray_directions_list = []
    target_pixel_values_list = []
    
    N, H, W, _ = images.shape
    for i in range(N):
        c2w = torch.tensor(poses[i], dtype=torch.float32)
        rays_o, rays_d = get_rays(H, W, focal, c2w)
        ray_origins_list.append(rays_o)
        ray_directions_list.append(rays_d)
        
        # images[i] is in (H, W, 3), already in [-1,1]
        target_pixels = torch.tensor(images[i].reshape(-1, 3), dtype=torch.float32, device=DEVICE)
        target_pixel_values_list.append(target_pixels)
    
    ray_origins = torch.cat(ray_origins_list, dim=0)
    ray_directions = torch.cat(ray_directions_list, dim=0)
    target_pixel_values = torch.cat(target_pixel_values_list, dim=0)
    
    all_rays = torch.cat([ray_origins, ray_directions, target_pixel_values], dim=-1)
    return all_rays
