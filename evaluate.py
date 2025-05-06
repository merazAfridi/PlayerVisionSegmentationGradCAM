import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset import FootballDataset, test_joint_transform
from models.unet_resnet50 import get_unet_resnet50
from utils.metrics import dice_coef, jaccard_index, pixel_accuracy

def main():
    # Define test set paths (update these paths to your local directories)
    test_image_dir  = r'C:\Users\mpower\Desktop\Football Players Segmentation\data\data\Testset\images'
    test_mask_dir   = r'C:\Users\mpower\Desktop\Football Players Segmentation\data\data\Testset\masks'

    # Create test dataset & loader 
    test_dataset = FootballDataset(test_image_dir, test_mask_dir, joint_transform=test_joint_transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    # Load the saved model checkpoint (update the checkpoint path if needed)
    checkpoint_path = r"C:\Users\mpower\Desktop\Football Players Segmentation\checkpoints\Unet_ResNet50.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Change map_location if using GPU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_unet_resnet50()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Evaluate on Test Set
    criterion = nn.BCEWithLogitsLoss()

    test_loss = 0.0
    test_acc = 0.0
    test_dice = 0.0
    test_jaccard = 0.0
    n_test_batches = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            test_loss += loss.item()
            test_dice += dice_coef(outputs, masks).item()
            test_jaccard += jaccard_index(outputs, masks).item()
            test_acc += pixel_accuracy(outputs, masks).item()
            n_test_batches += 1

    test_loss /= n_test_batches
    test_dice /= n_test_batches
    test_jaccard /= n_test_batches
    test_acc /= n_test_batches

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Dice: {test_dice:.4f}")
    print(f"Test Jaccard: {test_jaccard:.4f}")

    # Grad-CAM Analysis on one test sample
    test_iter = iter(test_loader)
    test_images, test_masks = next(test_iter)
    input_image_test = test_images[0:1].to(device)

    # Denormalize the test image for visualization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = input_image_test[0].cpu().numpy()  # shape: [3, H, W]
    img_denorm = img_norm * std[:, None, None] + mean[:, None, None]
    img_denorm = np.clip(img_denorm, 0, 1)
    denorm_img_np = np.transpose(img_denorm, (1, 2, 0))

    # Grad-CAM Analysis
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    target_layer = model.encoder.layer4[-1]  # Adjust if needed
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [lambda output: output.mean()]
    grayscale_cam = cam(input_tensor=input_image_test, targets=targets)[0]

    visualization = show_cam_on_image(denorm_img_np.astype(np.float32), grayscale_cam, use_rgb=True)

    # Get predicted mask for the test sample
    with torch.no_grad():
        output_test = model(input_image_test)
        pred_mask_test = torch.sigmoid(output_test)
        pred_mask_test = (pred_mask_test > 0.5).float().cpu().squeeze()

    # Plot the test sample, predicted mask, and Grad-CAM visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(denorm_img_np)
    axes[0].set_title("Test Image (De-Normalized)")
    axes[0].axis("off")
    axes[1].imshow(pred_mask_test, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")
    axes[2].imshow(visualization)
    axes[2].set_title("Grad-CAM")
    axes[2].axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()