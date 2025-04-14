import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Dataset import NeRFDataset
from Network import NeRFNetwork
from ray_sampling import get_rays, sample_rays
from Render import render_scene

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_loss(gt, pred):
    """
    Compute Mean Squared Error (MSE) and PSNR.
    PSNR = -10 * log10(MSE)
    """
    mse = torch.mean((gt - pred) ** 2)
    psnr = -10 * torch.log10(mse + 1e-10)
    return mse, psnr

def save_plots(loss_history, psnr_history, save_dir="logs"):
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot Loss per epoch
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, label="Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.legend()
    loss_plot_path = os.path.join(save_dir, "loss_per_epoch.png")
    plt.savefig(loss_plot_path)
    plt.close()
    
    # Plot PSNR per epoch
    plt.figure(figsize=(8, 6))
    plt.plot(psnr_history, label="PSNR per Epoch", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("Training PSNR")
    plt.legend()
    psnr_plot_path = os.path.join(save_dir, "psnr_per_epoch.png")
    plt.savefig(psnr_plot_path)
    plt.close()
    
    print(f"Saved loss plot at {loss_plot_path}")
    print(f"Saved PSNR plot at {psnr_plot_path}")

def train():
    # -------------------------------
    # Hyperparameters
    # -------------------------------
    num_epochs = 30000        # Increase this for better results
    batch_size = 1             # 1 image per iteration
    num_rays = 1024            # Number of rays sampled per image
    learning_rate = 5e-4       
    tn = 2.0                   
    tf = 6.0                   
    samples_per_ray = 192      
    
    root_dir = r"/home/smehta1/ComputerVision/NeRF/lego"
    train_dataset = NeRFDataset(root_dir=root_dir, split='train', img_wh=(800, 800))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = NeRFNetwork().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training on device: {DEVICE}")
    print(f"Total epochs: {num_epochs}")
    
    loss_history = []
    psnr_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_psnr = 0.0
        num_batches = 0
        
        for i, (img, pose, focal) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            
            # Because batch_size=1, extract the single image, pose, and focal.
            img = img[0].to(DEVICE)        # shape: (3, H, W)
            pose = pose[0].to(DEVICE)        # shape: (4, 4)
            if isinstance(focal, torch.Tensor):
                focal = focal.item()         # Convert to float if needed
            
            # Convert image from (C, H, W) to (H, W, 3)
            img = img.permute(1, 2, 0)
            H, W, _ = img.shape
            
            # Generate rays using camera-to-world pose and focal length.
            rays_o, rays_d = get_rays(H, W, focal, pose)
            
            # Ground truth pixels (flattened)
            gt_rgb = img.reshape(-1, 3)
            
            # Sample a subset of rays for training.
            sampled_rays_o, sampled_rays_d, sampled_gt_rgb = sample_rays(rays_o, rays_d, gt_rgb, num_rays)
            
            # Render the scene using hierarchical sampling.
            pred_rgb = render_scene(
                neural_field=model,
                origins=sampled_rays_o,
                directions=sampled_rays_d,
                tn=tn,
                tf=tf,
                samples=samples_per_ray,
                hierarchical=True,      # Enable hierarchical (importance) sampling
                N_importance=64,        # Number of fine samples
                clear_bg=False          # Use false with Tanh outputs in [-1, 1]
            )
            
            # Compute loss and PSNR.
            loss, psnr = get_loss(sampled_gt_rgb.to(DEVICE), pred_rgb)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_psnr += psnr.item()
            num_batches += 1
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.6f}, PSNR: {psnr.item():.2f}")
        
        avg_loss = epoch_loss / num_batches
        avg_psnr = epoch_psnr / num_batches
        loss_history.append(avg_loss)
        psnr_history.append(avg_psnr)
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.6f}, Avg PSNR: {avg_psnr:.2f}")
        
        # Save checkpoint and plots periodically.
        if (epoch + 1) % 1000 == 0:
            checkpoint_path = f"checkpoints/nerf_epoch_{epoch+1}.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            save_plots(loss_history, psnr_history, save_dir="logs")
    
    save_plots(loss_history, psnr_history, save_dir="logs")

if __name__ == "__main__":
    train()
