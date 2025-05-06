import segmentation_models_pytorch as smp

def get_unet_resnet50():
    model = smp.Unet(
        encoder_name="resnet50",          # use resnet50 as encoder
        encoder_weights="imagenet",         # pretrained on ImageNet
        in_channels=3,                      # RGB input
        classes=1                           # binary segmentation output
    )
    return model
