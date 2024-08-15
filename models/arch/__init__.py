# Add your custom network here
from .errnet import ERRNet


def errnet(in_channels, out_channels):
    return ERRNet(in_channels, out_channels)