import tensorflow
import torch
from pytorch_pretrained_biggan import BigGAN
from torchvision.models import resnet50
from networks import clip, mae
from typing import Text
import os

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

def load_discr(mode, device, path:Text='../models'):
    if mode == 'mae':
        # download this from https://www.kaggle.com/datasets/firstderivative/mae-pretrain-vit-base and store in ../models/mae
        discr = mae.load(os.path.join(path, 'mae/mae_pretrain_vit_base.pth')).to(device)
        #discr_layers = [f"blocks.{i}" for i in range(12)]
        discr_layers = []
        for name, layer in discr.named_modules():
            if  "mlp.act" in name:
                discr_layers.append(name)
                
    elif mode == "clip":
        #Note: jit = True not implemented
        discr, _ = clip.load("RN50", device=device, jit = False)
        discr_layers = [ "visual.layer1", "visual.layer2", "visual.layer3", "visual.layer4"]
        for p in discr.parameters(): 
            p.data = p.data.float()
    elif mode == "dino":
        discr = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50').to(device)
        for p in discr.parameters(): 
            p.data = p.data.float() 
        discr_layers = [ "layer1", "layer2", "layer3", "layer4"]
    elif mode in ['dino_vitb16', 'dino_vitb8']:
        discr = torch.hub.load('facebookresearch/dino:main', mode).to(device)
        for p in discr.parameters(): 
            p.data = p.data.float() 
        
        discr_layers = []
        for name, layer in discr.named_modules():
            if  "mlp.act" in name:
                discr_layers.append(name)
    elif mode == "resnet50":
        discr=resnet50(num_classes=1000,pretrained='imagenet').to(device)
        discr_layers = [ "layer1", "layer2", "layer3", "layer4"]
        for p in discr.parameters(): 
            p.data = p.data.float()
    else:
        raise NotImplementedError

    return discr, discr_layers