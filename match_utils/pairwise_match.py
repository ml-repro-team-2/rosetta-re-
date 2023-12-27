from match_utils import matching,dataset,nethook
import pickle
import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def get_BB(path,k=5):
    with open(os.path.join(path,'table.pkl'),"rb") as f:
        table=pickle.load(f)
    _,gan_matches=torch.topk(table,k=1,dim=1)
    _,discr_matches=torch.topk(table,k=k,dim=0)
    discr_matches=discr_matches.T
    BB_match={}
    for i in range(table.shape[0]):
        gan_matchesi=gan_matches[i]#these are the discriminators that match with the gan
        for matched_discr_unit in gan_matchesi:
            if i in discr_matches[matched_discr_unit]:
                BB_match[(i,matched_discr_unit.item())]=table[i,matched_discr_unit].item()
    return BB_match
def unit_layer_map(model_activs):
    ul_map={}
    base_unit=0
    for layer in model_activs.keys():
        for channel in range(model_activs[layer].shape[1]):
            ul_map[base_unit+channel]=(layer,channel)
        base_unit+=model_activs[layer].shape[1]
    return ul_map
def BB_viz(gan,gan_layers,discr,discr_layers,gan_mode,discr_mode,gan_stats,discr_stats,BB_matches,device,\
viz_size=10):
    gan.eval()
    discr.eval()

    #### hook layers for GAN if not already hooked
    if not isinstance(gan,nethook.InstrumentedModel):
        gan = nethook.InstrumentedModel(gan)
        gan.retain_layers(gan_layers)
    
    #### hook layers for discriminator if not already hooked
    if not isinstance(discr,nethook.InstrumentedModel):
      discr = nethook.InstrumentedModel(discr)
      discr.retain_layers(discr_layers)
    z,c = dataset.create_dataset(gan,gan_mode,1,1,5,device=device)
    z=z.float()
    c=c.float()
    
    with torch.no_grad():
        if gan_mode=='biggan':
            img=gan(z,c,1)
            img=(img+1)/2
        gan_activs=matching.store_activs(gan,gan_layers)   
        gan_unit_map=unit_layer_map(gan_activs)
        img = torch.nn.functional.interpolate(img, size = (224,224), mode = "bicubic")
        img = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
        _ = discr(img)            
        
        discr_activs = matching.store_activs(discr,discr_layers)
        gan_unit_map = unit_layer_map(gan_activs)
        discr_unit_map = unit_layer_map(discr_activs)
        img = img.squeeze().permute(1,2,0).contiguous().detach().cpu().numpy()
        plt.axis('off')
        plt.imshow(img)
        plt.show()
        for i,(gan_unit,discr_unit) in enumerate(BB_matches.keys()):
            ganlayer,ganunit = gan_unit_map[gan_unit]
            discrlayer,discrunit = discr_unit_map[discr_unit]
            gan_activ = matching.normalize(gan_activs[ganlayer],gan_stats[ganlayer])
            gan_activ = gan_activ[:,ganunit,:,:]
            discr_activ = matching.normalize(discr_activs[discrlayer],discr_stats[discrlayer])
            discr_activ = discr_activ[:,discrunit,:,:]
    
            map_size = max(gan_activ.shape[1],discr_activ.shape[1])
          
            gan_activ_new = torch.nn.Upsample(size=(map_size,map_size), mode='bilinear')(gan_activ.unsqueeze(1))
            discr_activ_new = torch.nn.Upsample(size=(map_size,map_size), mode='bilinear')(discr_activ.unsqueeze(1))             
        
            gan_activ_np=gan_activ_new.squeeze().detach().cpu().numpy()
            discr_activ_np=discr_activ_new.squeeze().detach().cpu().numpy()
            combined_activations = np.concatenate([gan_activ_np, discr_activ_np], axis=1)
            plt.axis('off')
            plt.imshow(combined_activations)
            plt.show()
            
            if i==viz_size-1:
                break
