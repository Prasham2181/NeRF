import os
import torch
import numpy as np
import cv2

from Dataset import NeRFDataset
from Network import NeRFNetwork
from ray_sampling import get_rays
from Render import render_scene

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def render_all_views(model, data_path, mode='test', H=800, W=800, batch_size=1024,
                     tn=2.0, tf=6.0, samples=192, output_dir='Logs/media'):
    """
    Render novel views for every image in the dataset and save them as separate files.
    
    Args:
        model: Trained NeRF model.
        data_path (str): Path to the dataset folder.
        mode (str): Dataset split to use (e.g., "test").
        H, W (int): Image dimensions.
        batch_size (int): Number of rays to process per batch.
        tn, tf (float): Near and far bounds along the ray.
        samples (int): Number of coarse samples per ray.
        output_dir (str): Folder to save the rendered views.
    """
    print("Loading dataset for all views...")
    dataset = NeRFDataset(root_dir=data_path, split=mode, img_wh=(H, W))
    focal = dataset[0][2]  # Focal length computed from the dataset
    os.makedirs(output_dir, exist_ok=True)
    num_images = len(dataset)
    
    for idx in range(num_images):
        print(f"Rendering view for image index: {idx}")
        # Get the image and its camera pose from the dataset.
        img, pose, focal_val = dataset[idx]
        pose = pose.to(DEVICE)
        # Use the focal from the dataset (already computed in NeRFDataset).
        focal_val = focal
        
        # Generate rays for the full image.
        rays_o, rays_d = get_rays(H, W, focal_val, pose)
        
        rendered_pixels = []
        for i in range(0, len(rays_o), batch_size):
            batch_rays_o = rays_o[i:i+batch_size].to(DEVICE)
            batch_rays_d = rays_d[i:i+batch_size].to(DEVICE)
            # Render the ray batch using hierarchical sampling.
            pred_rgb = render_scene(model, batch_rays_o, batch_rays_d,
                                    tn=tn, tf=tf, samples=samples,
                                    hierarchical=True, N_importance=64, clear_bg=False)
            rendered_pixels.append(pred_rgb.detach().cpu())
        rendered_pixels = torch.cat(rendered_pixels, dim=0)
        rendered_img = rendered_pixels.numpy().reshape(H, W, 3)
        # Map the network outputs from [-1, 1] to [0, 1]
        rendered_img_disp = (rendered_img + 1) / 2.0
        
        out_path = os.path.join(output_dir, f"view_{idx:04d}.png")
        cv2.imwrite(out_path, (rendered_img_disp * 255).astype(np.uint8))
        print(f"Saved rendered view for image {idx} at {out_path}")

if __name__ == "__main__":
    # Load the trained NeRF model.
    from Network import NeRFNetwork
    model = NeRFNetwork(Freq_L=10, direction_L=4).to(DEVICE)
    checkpoint_path = "checkpoints/nerf_epoch_1000.pth"  # Adjust as needed
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    # Render novel views for all images in the test split.
    render_all_views(model, data_path="Phase2/Nerf_Dataset/nerf_synthetic/lego/", mode="test", H=800, W=800)
