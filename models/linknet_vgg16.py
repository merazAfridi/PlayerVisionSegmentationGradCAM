import segmentation_models_pytorch as smp

def get_linknet_vgg16():
    model = smp.Linknet(
        encoder_name="vgg16",         # use vgg16 as encoder
        encoder_weights="imagenet",     # pretrained on ImageNet
        in_channels=3,                # RGB input
        classes=1                     # binary segmentation output
    )
    return model
