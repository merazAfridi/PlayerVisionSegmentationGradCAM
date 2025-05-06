import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FootballDataset(Dataset):
    def __init__(self, image_dir, mask_dir, joint_transform=None):
        """
        Args:
            image_dir (str): Directory with images.
            mask_dir (str): Directory with masks.
            joint_transform (callable, optional): Augmentation/Preprocessing to apply jointly.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = sorted(os.listdir(image_dir))
        self.mask_names  = sorted(os.listdir(mask_dir))
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path  = os.path.join(self.mask_dir, self.mask_names[idx])
        image = Image.open(image_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")  # grayscale mask

        # Convert to NumPy arrays
        image_np = np.array(image)
        mask_np  = np.array(mask)

        if self.joint_transform is not None:
            augmented = self.joint_transform(image=image_np, mask=mask_np)
            image = augmented["image"]
            mask  = augmented["mask"]

        # Ensure mask has a channel dimension (1, H, W)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        # Convert mask to float and scale to [0, 1]
        mask = mask.float() / 255.0
        
        return image, mask

# Augmentation / Preprocessing Transforms
train_joint_transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'mask': 'mask'}, is_check_shapes=False)

val_joint_transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], additional_targets={'mask': 'mask'}, is_check_shapes=False)

test_joint_transform = val_joint_transform
