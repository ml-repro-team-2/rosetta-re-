import tensorflow
import torch
from pytorch_pretrained_biggan import BigGAN
from torchvision.models import resnet50
from typing import Text

def load_gan(mode,device='cpu',path: Text='.'):
    if mode=="biggan":
        gan = BigGAN.from_pretrained('biggan-deep-256').to(device)
        gan_layers = []
        for name, layer in gan.named_modules():
            if "conv" in name:
                gan_layers.append(name)
    
    elif mode=="stylegan-lsun_horse":
        pass
    return gan,gan_layers
def load_discr(mode, device, path:Text='.'):
    if mode == "resnet50":
        discr=resnet50(num_classes=1000,pretrained='imagenet').to(device)
        discr_layers = [ "layer1", "layer2", "layer3", "layer4"]
        for p in discr.parameters(): 
            p.data = p.data.float()
    return discr, discr_layers