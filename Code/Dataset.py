import torch
import numpy as np
import os
import json
from PIL import Image

class NeRFDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800)):
        """
        Args:
            root_dir (str): Path to the dataset directory.
            split (str): Which split to load (train/val/test).
            img_wh (tuple): (width, height) for resizing images.
        """
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh

        # Load the camera transformations from JSON
        json_path = os.path.join(root_dir, f'transforms_{split}.json')
        with open(json_path, 'r') as f:
            self.meta = json.load(f)

        # Extract camera field of view and compute focal length
        self.camera_angle_x = self.meta['camera_angle_x']
        # Convert to radians if necessary
        if self.camera_angle_x > np.pi:
            self.camera_angle_x = np.deg2rad(self.camera_angle_x)

        self.focal = 0.5 * img_wh[0] / np.tan(0.5 * self.camera_angle_x)

        self.image_paths = []
        self.poses = []

        for frame in self.meta['frames']:
            file_path = frame['file_path']
            img_path = os.path.join(root_dir, f"{file_path}.png")
            self.image_paths.append(img_path)

            transform_matrix = np.array(frame['transform_matrix'], dtype=np.float32)  # 4x4 camera pose
            self.poses.append(transform_matrix)

        self.poses = torch.from_numpy(np.array(self.poses, dtype=np.float32))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns:
            img (Tensor): Shape (3, H, W), normalized to [-1, 1].
            pose (Tensor): Shape (4, 4).
            focal (float): The camera focal length.
        """
        img = Image.open(self.image_paths[idx])
        img = img.resize(self.img_wh, Image.LANCZOS)
        # Normalize from [0,255] to [-1,1]
        img = np.array(img, dtype=np.float32) / 127.5 - 1.0

        # If image has an alpha channel, discard it
        if img.shape[-1] == 4:
            img = img[..., :3]

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        pose = self.poses[idx]
        return img, pose, self.focal
