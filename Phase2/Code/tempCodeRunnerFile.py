import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import your modules (adjust the import paths as needed)
from Dataset import NeRFDataset
from Network import NeRFNetwork
from ray_sampling import get_rays, sample_rays
from Render import render_scene

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # -------------------------------
    # Hyperparameters and settings
    # -------------------------------
    num_epochs = 20            # Total number of epochs
    batch_size = 1             # Process one image per iteration
    num_rays = 1024            # Number of rays sampled per image
    learning_rate = 5e-4       # Learning rate for Adam optimizer
    tn = 2.0                   # Near bound along rays
    tf = 6.0                   # Far bound along rays
    samples_per_ray = 192      # Number of samples per ray
    
    # Modify this path to point to your dataset folder.
    root_dir = r"Phase2\Nerf_Dataset\nerf_synthetic\lego"  # Example path
    train_dataset = NeRFDataset(root_dir=root_dir, split='train', img_wh=(800, 800))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # -------------------------------
    # Initialize the NeRF model and optimizer
    # -------------------------------
    model = NeRFNetwork().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # -------------------------------
    # Training loop
    # -------------------------------
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # Iterate over the training dataset (each iteration processes one image)
        for i, (img, pose, focal) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            
            # Since batch_size=1, extract the single image, pose, and focal length.
            # Note: Dataset returns image in (C, H, W), so convert to (H, W, 3)
            img = img[0].to(DEVICE)        # shape: (3, H, W)
            pose = pose[0].to(DEVICE)      # shape: (4, 4)
            if isinstance(focal, torch.Tensor):
                focal = focal.item()       # Convert focal to float if needed
            
            # Convert image from (C, H, W) to (H, W, 3)
            img = img.permute(1, 2, 0)
            H, W, _ = img.shape
            
            # Generate rays for the image using the camera-to-world pose and focal length.
            rays_o, rays_d = get_rays(H, W, focal, pose)
            
            # Reshape ground truth image pixels to (H*W, 3)
            gt_rgb = img.reshape(-1, 3)
            
            # Sample a subset of rays (and corresponding ground truth colors)
            sampled_rays_o, sampled_rays_d, sampled_gt_rgb = sample_rays(rays_o, rays_d, gt_rgb, num_rays)
            
            # Render the scene for the sampled rays using the current model.
            pred_rgb = render_scene(
                neural_field=model,
                origins=sampled_rays_o,
                directions=sampled_rays_d,
                tn=tn,
                tf=tf,
                samples=samples_per_ray,
                clear_bg=True
            )
            
            # Compute the mean squared error between the predicted and ground truth colors.
            loss = F.mse_loss(pred_rgb, sampled_gt_rgb)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
        
        # -------------------------------
        # Optionally: Save checkpoints periodically.
        # -------------------------------
        if (epoch + 1) % 100 == 0:
            checkpoint_path = f"checkpoints/nerf_epoch_{epoch+1}.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
        # -------------------------------
        # Optionally: Visualize or log rendered images.
        # -------------------------------
        # (You might render a full image using render_scene over all rays and display/save it.)
    
if __name__ == "__main__":
    train()
