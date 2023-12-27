import torch
import torchvision
from scipy.stats import truncnorm

def truncate_noise(size,truncation):
    truncated_noise=truncnorm.rvs(-truncation,truncation,size=size)#
    return torch.tensor(truncated_noise)
def create_dataset(gan,ganmode,batch_size,epochs,classidx,device):
    '''create dataset for GAN'''
    if ganmode=="biggan":
        z_dataset=truncate_noise((batch_size*epochs,128),1).to(device)
        c_dataset=torch.zeros((batch_size*epochs,1000)).to(device)
        c_dataset[:,classidx]=1
    return z_dataset,c_dataset