import os
import time
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from Dataset import NeRFDataset
from Network import NeRFNetwork
from ray_sampling import get_rays, sample_rays, generateBatch
from Render import render_scene

def get_loss(gt, pred):
    """
    Compute Mean Squared Error (MSE) and PSNR.
    PSNR = -10 * log10(MSE)
    
    Args:
        gt (torch.Tensor): Ground truth RGB values
        pred (torch.Tensor): Predicted RGB values
    
    Returns:
        tuple: (mse, psnr) metrics
    """
    mse = torch.mean((gt - pred) ** 2)
    psnr = -10 * torch.log10(mse + 1e-10)
    return mse, psnr

def val(model, data_path, epoch, mode='val', batch_size=1024, tn=2.0, tf=6.0, samples=192, 
        log_every=100, verbose=False, fixed_indices=None, device=None):
    """
    Validation routine for the NeRF model.
    
    Args:
        model (NeRFNetwork): The neural network model
        data_path (str): Path to the dataset directory
        epoch (int): Current epoch number for logging
        mode (str): Dataset split to use ('train', 'val', or 'test')
        batch_size (int): Number of rays to process at once
        tn (float): Near bound for ray sampling
        tf (float): Far bound for ray sampling
        samples (int): Number of samples per ray
        log_every (int): Log interval for batch progress
        verbose (bool): Whether to print detailed progress
        fixed_indices (list): Optional list of specific image indices to use
        device (torch.device): Computing device
    
    Returns:
        float: Average loss value across all validation batches
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading {mode} data from {data_path}...")
    dataset = NeRFDataset(root_dir=data_path, split=mode, img_wh=(800, 800))
    focal = dataset[0][2]
    
    # Load images and poses
    images = []
    poses = []
    for img, pose, _ in dataset:
        images.append(img.permute(1, 2, 0).numpy())  # (H, W, 3) in [-1, 1]
        poses.append(pose.numpy())
    images = np.array(images)
    poses = np.array(poses)
    
    # Select validation images
    num_val_imgs = 4
    if fixed_indices is not None:
        indices = fixed_indices
    else:
        indices = np.random.choice(np.arange(len(images)), num_val_imgs, replace=False)
    
    sel_images = images[indices]
    sel_poses = poses[indices]
    
    print(f"Validation Indices: {indices}")
    
    # Generate rays for the selected images
    try:
        all_rays = generateBatch(sel_images, sel_poses, focal)  # shape: (total_pixels, 9)
        print(f"Generated rays shape: {all_rays.shape if hasattr(all_rays, 'shape') else 'unknown'}")
        print(f"Rays type: {type(all_rays)}")
        
        # Ensure all_rays is a numpy array if it's a tensor
        if isinstance(all_rays, torch.Tensor):
            all_rays = all_rays.detach().cpu().numpy()
            print("Converted tensor rays to numpy array")
    except Exception as e:
        print(f"Error during ray generation: {e}")
        raise
    
    # Process in batches
    num_batches = len(all_rays) // batch_size + (1 if len(all_rays) % batch_size != 0 else 0)
    
    avg_loss = 0.0
    avg_psnr = 0.0
    
    try:
        for i in tqdm(range(num_batches), desc=f"Validating epoch {epoch}"):
            # Use sequential batches for reproducibility
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(all_rays))
            batch = all_rays[start_idx:end_idx]
            
            # Check if batch is already a tensor
            if isinstance(batch, torch.Tensor):
                # If it's already a tensor, just move it to the right device
                rays_origin = batch[:, :3].to(dtype=torch.float32, device=device)
                rays_direction = batch[:, 3:6].to(dtype=torch.float32, device=device)
                target_pixels = batch[:, 6:].to(dtype=torch.float32, device=device)
            else:
                # If it's a numpy array, convert to tensor
                rays_origin = torch.from_numpy(batch[:, :3]).to(dtype=torch.float32, device=device)
                rays_direction = torch.from_numpy(batch[:, 3:6]).to(dtype=torch.float32, device=device)
                target_pixels = torch.from_numpy(batch[:, 6:]).to(dtype=torch.float32, device=device)
            
            # Render with hierarchical sampling
            with torch.no_grad():
                pred_pixels = render_scene(model, rays_origin, rays_direction,
                                          tn=tn, tf=tf, samples=samples,
                                          hierarchical=True, N_importance=64, clear_bg=False)
            
            loss, psnr = get_loss(target_pixels, pred_pixels)
            avg_loss += loss.item()
            avg_psnr += psnr.item()
            
            if verbose and i % log_every == 0:
                print(f'Batch {i}/{num_batches}, Loss: {loss.item():.6f}, PSNR: {psnr.item():.2f}')
            
            # Clear cache periodically
            if torch.cuda.is_available() and i % 10 == 0:
                torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("\nValidation interrupted. Calculating metrics with batches processed so far...")
        if i > 0:  # Only adjust if at least one batch was processed
            num_batches = i
    
    # Calculate final metrics
    if num_batches > 0:
        avg_loss /= num_batches
        avg_psnr /= num_batches
        print(f"Epoch {epoch} Validation: Avg Loss = {avg_loss:.6f}, Avg PSNR = {avg_psnr:.2f}")
    else:
        print("No batches were processed. Cannot calculate metrics.")
        avg_loss = float('nan')
    
    return avg_loss

def test(model, data_path, mode='test', image_idx=0, batch_size=1024, tn=2.0, tf=6.0, samples=192,
         H=800, W=800, log_dir='Logs', img_name=None, device=None):
    """
    Testing routine for a specific image.
    
    Args:
        model (NeRFNetwork): The neural network model
        data_path (str): Path to the dataset directory
        mode (str): Dataset split to use
        image_idx (int): Index of the image to test
        batch_size (int): Number of rays to process at once
        tn (float): Near bound for ray sampling
        tf (float): Far bound for ray sampling
        samples (int): Number of samples per ray
        H (int): Image height
        W (int): Image width
        log_dir (str): Directory to save results
        img_name (str): Name for the saved image
        device (torch.device): Computing device
    
    Returns:
        tuple: (psnr, ssim) metrics for the rendered image
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if img_name is None:
        img_name = f"test_img_{image_idx}.png"
    
    print(f"Loading {mode} data for image {image_idx} from {data_path}...")
    dataset = NeRFDataset(root_dir=data_path, split=mode, img_wh=(H, W))
    focal = dataset[0][2]
    
    # Load images and poses
    images = []
    poses = []
    for img, pose, _ in dataset:
        images.append(img.permute(1, 2, 0).numpy())  # (H, W, 3) in [-1, 1]
        poses.append(pose.numpy())
    images = np.array(images)
    poses = np.array(poses)
    
    test_img = images[image_idx]
    test_pose = poses[image_idx]
    
    # Generate rays for the full image
    rays_o, rays_d = get_rays(H, W, focal, torch.tensor(test_pose, dtype=torch.float32))
    gt_rgb = torch.tensor(test_img.reshape(-1, 3), dtype=torch.float32, device=device)
    
    # Render in batches
    rendered_pixels = []
    num_batches = len(rays_o) // batch_size + (1 if len(rays_o) % batch_size != 0 else 0)
    
    try:
        for i in tqdm(range(num_batches), desc=f"Rendering test image {image_idx}"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(rays_o))
            
            batch_rays_o = rays_o[start_idx:end_idx].to(device)
            batch_rays_d = rays_d[start_idx:end_idx].to(device)
            
            with torch.no_grad():
                pred_rgb = render_scene(model, batch_rays_o, batch_rays_d,
                                       tn=tn, tf=tf, samples=samples,
                                       hierarchical=True, N_importance=64, clear_bg=False)
            
            rendered_pixels.append(pred_rgb.detach().cpu())
            
            # Clear cache periodically
            if torch.cuda.is_available() and i % 10 == 0:
                torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("\nRendering interrupted. Using partially rendered image...")
    
    # Combine all batches
    rendered_pixels = torch.cat(rendered_pixels, dim=0)
    rendered_img = rendered_pixels.numpy().reshape(H, W, 3)
    
    # Map from [-1, 1] to [0, 1] for display/saving
    rendered_img_disp = (rendered_img + 1) / 2.0
    test_img_disp = (test_img + 1) / 2.0
    
    # Calculate metrics
    mse = np.mean((rendered_img - test_img) ** 2)
    psnr = -10 * np.log10(mse + 1e-10)
    
    # Calculate SSIM per channel and average
    ssim_r = ssim((rendered_img_disp[:,:,0]*255).astype(np.uint8),
                  (test_img_disp[:,:,0]*255).astype(np.uint8),
                  data_range=255)
    ssim_g = ssim((rendered_img_disp[:,:,1]*255).astype(np.uint8),
                  (test_img_disp[:,:,1]*255).astype(np.uint8),
                  data_range=255)
    ssim_b = ssim((rendered_img_disp[:,:,2]*255).astype(np.uint8),
                  (test_img_disp[:,:,2]*255).astype(np.uint8),
                  data_range=255)
    ssim_val = (ssim_r + ssim_g + ssim_b) / 3
    
    print(f"Test Image {image_idx}: PSNR: {psnr:.2f}, SSIM: {ssim_val:.4f}")
    
    # Save rendered image
    os.makedirs(os.path.join(log_dir, "media"), exist_ok=True)
    save_path = os.path.join(log_dir, "media", img_name)
    
    # OpenCV expects BGR format for saving
    cv2_img = cv2.cvtColor((rendered_img_disp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, cv2_img)
    print(f"Saved rendered test image at {save_path}")
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(test_img_disp)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    
    axes[1].imshow(rendered_img_disp)
    axes[1].set_title(f"Rendered (PSNR: {psnr:.2f}, SSIM: {ssim_val:.4f})")
    axes[1].axis("off")
    
    plt.tight_layout()
    comparison_path = os.path.join(log_dir, "media", f"comparison_{image_idx}.png")
    plt.savefig(comparison_path)
    plt.close()
    
    return psnr, ssim_val

def test_single_image(model, tn=2.0, tf=6.0, samples=192, batch_size=256, H=800, W=800, 
                     log_dir='Logs', theta=45, phi=-30, radius=4, device=None):
    """
    Render a single novel view using a generated camera pose.
    
    Args:
        model (NeRFNetwork): The neural network model
        tn (float): Near bound for ray sampling
        tf (float): Far bound for ray sampling
        samples (int): Number of samples per ray
        batch_size (int): Number of rays to process at once
        H (int): Image height
        W (int): Image width
        log_dir (str): Directory to save results
        theta (float): Angle for camera position (degrees)
        phi (float): Angle for camera position (degrees)
        radius (float): Distance of camera from origin
        device (torch.device): Computing device
    
    Returns:
        np.ndarray: The rendered image
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    focal = 555.5555  # Focal length

    def pose_spherical(theta, phi, radius):
        """Generate camera pose from spherical coordinates."""
        c2w = np.eye(4, dtype=np.float32)
        c2w[0,3] = radius * np.sin(np.radians(theta)) * np.cos(np.radians(phi))
        c2w[1,3] = radius * np.sin(np.radians(theta)) * np.sin(np.radians(phi))
        c2w[2,3] = radius * np.cos(np.radians(theta))
        return c2w

    # Generate camera pose
    c2w = pose_spherical(theta, phi, radius)
    print(f"Rendering novel view at theta={theta}°, phi={phi}°, radius={radius}")
    
    # Generate rays
    rays_o, rays_d = get_rays(H, W, focal, torch.tensor(c2w, dtype=torch.float32))
    
    # Render in batches
    rendered_pixels = []
    num_batches = len(rays_o) // batch_size + (1 if len(rays_o) % batch_size != 0 else 0)
    
    try:
        for i in tqdm(range(num_batches), desc="Rendering novel view"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(rays_o))
            
            batch_rays_o = rays_o[start_idx:end_idx].to(device)
            batch_rays_d = rays_d[start_idx:end_idx].to(device)
            
            with torch.no_grad():
                pred_rgb = render_scene(model, batch_rays_o, batch_rays_d,
                                      tn=tn, tf=tf, samples=samples,
                                      hierarchical=True, N_importance=64, clear_bg=False)
            
            rendered_pixels.append(pred_rgb.detach().cpu())
            
            # Clear cache periodically
            if torch.cuda.is_available() and i % 10 == 0:
                torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("\nRendering interrupted. Using partially rendered image...")
    
    # Combine all batches
    rendered_pixels = torch.cat(rendered_pixels, dim=0)
    rendered_img = rendered_pixels.numpy().reshape(H, W, 3)
    
    # Map from [-1, 1] to [0, 1] for display/saving
    rendered_img_disp = (rendered_img + 1) / 2.0
    
    # Save rendered image
    os.makedirs(os.path.join(log_dir, "media"), exist_ok=True)
    save_path = os.path.join(log_dir, "media", f"novel_view_t{theta}_p{phi}_r{radius}.png")
    
    # OpenCV expects BGR format for saving
    cv2_img = cv2.cvtColor((rendered_img_disp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, cv2_img)
    print(f"Saved novel view image at {save_path}")
    
    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(rendered_img_disp)
    plt.title(f"Novel View (θ={theta}°, φ={phi}°, r={radius})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "media", f"novel_view_plot_t{theta}_p{phi}_r{radius}.png"))
    plt.close()
    
    return rendered_img_disp

def create_spiral_rendering(model, n_frames=60, tn=2.0, tf=6.0, samples=192, batch_size=256, 
                           H=800, W=800, log_dir='Logs', radius=4, device=None):
    """
    Create a spiral path rendering for video creation.
    
    Args:
        model (NeRFNetwork): The neural network model
        n_frames (int): Number of frames to render
        tn (float): Near bound for ray sampling
        tf (float): Far bound for ray sampling
        samples (int): Number of samples per ray
        batch_size (int): Number of rays to process at once
        H (int): Image height
        W (int): Image width
        log_dir (str): Directory to save results
        radius (float): Distance of camera from origin
        device (torch.device): Computing device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create directory for video frames
    video_dir = os.path.join(log_dir, "media", "spiral")
    os.makedirs(video_dir, exist_ok=True)
    
    print(f"Rendering {n_frames} frames for spiral path video...")
    
    for i in tqdm(range(n_frames), desc="Rendering spiral frames"):
        # Calculate camera position along spiral path
        theta = 180 * (i / n_frames)  # Rotate around
        phi = -30 + 20 * np.sin(2 * np.pi * i / n_frames)  # Up and down
        r = radius - 0.5 * np.cos(2 * np.pi * i / n_frames)  # In and out
        
        # Render the frame
        frame = test_single_image(
            model, tn=tn, tf=tf, samples=samples, batch_size=batch_size,
            H=H, W=W, log_dir=log_dir, theta=theta, phi=phi, radius=r,
            device=device
        )
        
        # Save the frame
        frame_path = os.path.join(video_dir, f"frame_{i:04d}.png")
        cv2_img = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_path, cv2_img)
    
    print(f"Rendered {n_frames} frames saved to {video_dir}")
    print("To create a video from these frames, you can use ffmpeg:")
    print(f"ffmpeg -framerate 30 -i {video_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {log_dir}/media/spiral_video.mp4")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NeRF Validation and Testing')
    
    # Model and data paths
    parser.add_argument('--checkpoint', type=str, 
                        default="/home/smehta1/ComputerVision/NeRF/Phase2/checkpoints/nerf_epoch_1000.pth", 
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, 
                        default="/home/smehta1/ComputerVision/NeRF/lego", 
                        help='Path to dataset directory')
    parser.add_argument('--log_dir', type=str, default='Logs', 
                        help='Directory for saving logs and media')
    
    # Validation options
    parser.add_argument('--mode', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset split to use')
    parser.add_argument('--batch_size', type=int, default=1024, 
                        help='Batch size for ray processing')
    parser.add_argument('--tn', type=float, default=2.0, 
                        help='Near bound for ray sampling')
    parser.add_argument('--tf', type=float, default=6.0, 
                        help='Far bound for ray sampling')
    parser.add_argument('--samples', type=int, default=192, 
                        help='Number of samples per ray')
    
    # Image options
    parser.add_argument('--img_size', type=int, default=800, 
                        help='Image size (assumed square)')
    parser.add_argument('--test_idx', type=int, default=0, 
                        help='Index of test image to render')
    
    # Novel view options
    parser.add_argument('--novel_view', action='store_true', 
                        help='Render a novel view')
    parser.add_argument('--theta', type=float, default=45, 
                        help='Theta angle for novel view')
    parser.add_argument('--phi', type=float, default=-30, 
                        help='Phi angle for novel view')
    parser.add_argument('--radius', type=float, default=4, 
                        help='Radius for novel view')
    
    # Spiral video options
    parser.add_argument('--create_spiral', action='store_true', 
                        help='Create a spiral path rendering')
    parser.add_argument('--n_frames', type=int, default=60, 
                        help='Number of frames for spiral rendering')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = NeRFNetwork(Freq_L=10, direction_L=4).to(device)
    
    # Use weights_only=True to avoid security warning
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    
    try:
        # Run validation
        print("\n=== Running Validation ===")
        val_loss = val(
            model, data_path=args.data_path, epoch=0, mode=args.mode,
            batch_size=args.batch_size, tn=args.tn, tf=args.tf, samples=args.samples,
            verbose=True, device=device
        )
        print(f"Average Validation Loss: {val_loss:.6f}")
        
        # Run test on specific image
        print("\n=== Testing Specific Image ===")
        test(
            model, data_path=args.data_path, mode=args.mode, image_idx=args.test_idx,
            batch_size=args.batch_size, tn=args.tn, tf=args.tf, samples=args.samples,
            H=args.img_size, W=args.img_size, log_dir=args.log_dir, device=device
        )
        
        # Render novel view if requested
        if args.novel_view:
            print("\n=== Rendering Novel View ===")
            test_single_image(
                model, tn=args.tn, tf=args.tf, samples=args.samples,
                batch_size=args.batch_size, H=args.img_size, W=args.img_size,
                log_dir=args.log_dir, theta=args.theta, phi=args.phi, radius=args.radius,
                device=device
            )
        
        # Create spiral rendering if requested
        if args.create_spiral:
            print("\n=== Creating Spiral Rendering ===")
            create_spiral_rendering(
                model, n_frames=args.n_frames, tn=args.tn, tf=args.tf, samples=args.samples,
                batch_size=args.batch_size, H=args.img_size, W=args.img_size,
                log_dir=args.log_dir, radius=args.radius, device=device
            )
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Process complete.")

if __name__ == "__main__":
    main()