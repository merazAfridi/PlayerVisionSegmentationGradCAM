import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import multiprocessing

from dataset import FootballDataset, train_joint_transform, val_joint_transform, test_joint_transform
from utils.metrics import dice_coef, jaccard_index, pixel_accuracy

def denormalize(image, mean, std):
    "denormalize img using: mean & std"
    mean = np.array(mean)
    std = np.array(std)
    image = image * std + mean
    return np.clip(image, 0, 1)

def main():
    
    train_image_dir = r'C:\Users\mpower\Desktop\Football Players Segmentation\data\data\Trainset\images'
    train_mask_dir = r'C:\Users\mpower\Desktop\Football Players Segmentation\data\data\Trainset\masks'
    test_image_dir = r'C:\Users\mpower\Desktop\Football Players Segmentation\data\data\Testset\images'
    test_mask_dir = r'C:\Users\mpower\Desktop\Football Players Segmentation\data\data\Testset\masks'



    #creatig dataset instances, and Split it
    full_dataset = FootballDataset(train_image_dir, train_mask_dir, joint_transform=train_joint_transform)
    total_samples = len(full_dataset)
    split_index = int(0.9 * total_samples)
    indices = list(range(total_samples))
    train_indices = indices[:split_index]
    val_indices   = indices[split_index:]

    #creating separate dataset by val transform for val
    full_dataset_val = FootballDataset(train_image_dir, train_mask_dir, joint_transform=val_joint_transform)
    train_subset = Subset(full_dataset, train_indices)
    val_subset   = Subset(full_dataset_val, val_indices)

    #dataloader
    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=2)

    #creating test data set
    test_dataset = FootballDataset(test_image_dir, test_mask_dir, joint_transform=test_joint_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)


    #viisualize
    sample_image, sample_mask = full_dataset[0]
    sample_image_np = sample_image.cpu().numpy().transpose(1, 2, 0)  #convert to HWC format
    sample_image_np = denormalize(sample_image_np, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    #specify vmin and vmax to ensure data 
    #is interpreted in the range [0,1]
    plt.imshow(sample_image_np, vmin=0, vmax=1)
    plt.title("Augmented & Resized Image (512x512)")
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(sample_mask.squeeze(), cmap='gray')
    plt.title("Mask (512x512)")
    plt.axis('off')
    plt.show()

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="resnet50",      #encoder= resnet50
        encoder_weights="imagenet",         
        in_channels=3,             #RGB image
        classes=1,     #binary segm output
    )
    model = model.to(device)

    # Loss, Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    total_training_time = 0.0
    checkpoint_dir = r'C:\Users\mpower\Desktop\Football Players Segmentation\checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_dices = []
    val_dices = []
    train_jaccards = []
    val_jaccards = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        #training
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_jaccard = 0.0
        train_acc = 0.0
        n_train_batches = 0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)  #shape=[ B, 1, 512, 512 ] 
                                         #with values [0,1]
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice_coef(outputs, masks).item()
            train_jaccard += jaccard_index(outputs, masks).item()
            train_acc += pixel_accuracy(outputs, masks).item()
            n_train_batches += 1
            

            print(f"Batch {n_train_batches} - Mask min: {masks.min().item()}, Mask max: {masks.max().item()}")
            print(f"Batch {n_train_batches} - Output min: {outputs.min().item()}, Output max: {outputs.max().item()}")

        train_loss /= n_train_batches
        train_dice /= n_train_batches
        train_jaccard /= n_train_batches
        train_acc /= n_train_batches

        #Val
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_jaccard = 0.0
        val_acc = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice_coef(outputs, masks).item()
                val_jaccard += jaccard_index(outputs, masks).item()
                val_acc += pixel_accuracy(outputs, masks).item()
                n_val_batches += 1
                
                print(f"Validation Batch {n_val_batches} - Mask min: {masks.min().item()}, Mask max: {masks.max().item()}")
                print(f"Validation Batch {n_val_batches} - Output min: {outputs.min().item()}, Output max: {outputs.max().item()}")

        val_loss /= n_val_batches
        val_dice /= n_val_batches
        val_jaccard /= n_val_batches
        val_acc /= n_val_batches

        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time

        #appending metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        train_jaccards.append(train_jaccard)
        val_jaccards.append(val_jaccard)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Dice: {train_dice:.4f}, Train Jaccard: {train_jaccard:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Dice: {val_dice:.4f}, Val Jaccard: {val_jaccard:.4f} | "
              f"Time: {epoch_time:.2f}s")

    #save final model
    checkpoint_path = os.path.join(checkpoint_dir, 'Unet_ResNet50.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_dice': train_dice,
        'val_dice': val_dice,
        'train_jaccard': train_jaccard,
        'val_jaccard': val_jaccard,
        'train_acc': train_acc,
        'val_acc': val_acc
    }, checkpoint_path)

    print(f"Final model saved at {checkpoint_path}")
    print(f"Total Training Time: {total_training_time:.2f}s")

    #loss plot
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 10))
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training Loss')
    plt.plot(epochs, val_losses, 'r', label='Val Loss')
    plt.title('Training and Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Val Accuracy')
    plt.title('Training and Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    #dice coef
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_dices, 'b', label='Training Dice')
    plt.plot(epochs, val_dices, 'r', label='Val Dice')
    plt.title('Training and Val Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    #Jaccard index
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_jaccards, 'b', label='Training Jaccard')
    plt.plot(epochs, val_jaccards, 'r', label='Val Jaccard')
    plt.title('Training and Val Jaccard Index')
    plt.xlabel('Epochs')
    plt.ylabel('Jaccard Index')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()