from match_utils import nethook, dataset, stats
import torch
import torchvision
import numpy as np
import pickle
import os
import tqdm
def normalize(activation,stats_table):
    eps=1e-5
    norm_input=(activation-stats_table[0])/(stats_table[1]+eps)
    return norm_input

def store_activs(model, layernames):
    activs={}
    for layer in layernames:
        activs[layer]=model._retrieve_retained(layer,clear=True)
    return activs

def dict_layers(activs,layernames):
    layer_channels={}
    for layer in layernames:
        layer_channels[layer]=activs[layer].shape[1]
    return layer_channels

def save_array(array, filename):
    open_file = open(filename, "wb")
    pickle.dump(array, open_file)
    open_file.close()

def activ_match_gan(gan, gan_layers, discr,discr_layers, gan_mode, discr_mode,\
                    dataset, epochs, batch_size, save_path, device):
    gan.eval()
    discr.eval()

    gan=nethook.InstrumentedModel(gan)
    gan.retain_layers(gan_layers)

    discr=nethook.InstrumentedModel(discr)
    discr.retain_layers(discr_layers)



    gan_stats_table,discr_stats_table=stats.get_mean_std(gan, gan_layers, discr, discr_layers, gan_mode, discr_mode, dataset, epochs, batch_size, device)
    save_array(gan_stats_table, os.path.join(save_path, "gan_stats.pkl"))
    save_array(discr_stats_table, os.path.join(save_path, "discr_stats.pkl"))
    global final_match_table
    for iteration in tqdm.trange(0,epochs):
        with torch.no_grad():

            z = dataset[0][iteration*batch_size: (iteration+1)*batch_size ].float()
            c = dataset[1][iteration*batch_size: (iteration+1)*batch_size ].float()
            
            
            if gan_mode == "biggan":
                img = gan(z,c,1)
                img = (img+1)/2
            del z
            del c
            gan_activs=store_activs(gan,gan_layers)

            #clip condition
            img = torch.nn.functional.interpolate(img, size = (224,224), mode = "bicubic")
            img = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            _ = discr(img)
            del img
            discr_activs=store_activs(discr,discr_layers)
            if iteration==0:
                gan_channel=dict_layers(gan_activs,gan_layers)
                discr_channel=dict_layers(discr_activs,discr_layers)

                num_gan_activs = sum(gan_channel.values())
                num_discr_activs = sum(discr_channel.values())
                final_match_table = torch.zeros((num_gan_activs, num_discr_activs)).to(device)

            i=0
            for gan_layer in gan_layers:
                gan_activ=normalize(gan_activs[gan_layer],gan_stats_table[gan_layer])
                
                j=0
                for discr_layer in discr_layers:
                    discr_activ=normalize(discr_activs[discr_layer],discr_stats_table[discr_layer])
                    
                    map_size = max((gan_activ.shape[2], discr_activ.shape[2]))
                    gan_activ_new = torch.nn.Upsample(size=(map_size,map_size), mode='bilinear')(gan_activ)
                    discr_activ_new = torch.nn.Upsample(size=(map_size,map_size), mode='bilinear')(discr_activ)  
                    
                    final_match_table[i:i+gan_activ.shape[1],j:j+discr_activ.shape[1]]+= torch.einsum('aixy,ajxy->ij', gan_activ_new,discr_activ_new)
                    final_match_table[i:i+gan_activ.shape[1],j:j+discr_activ.shape[1]]/= map_size*map_size
                    
                    

                    j+=discr_activ.shape[1]
                    del gan_activ_new
                    del discr_activ_new
    

                del gan_activs[gan_layer]
                
                i+=gan_activ.shape[1]
            del discr_activs
    final_match_table /= epochs*batch_size
    save_array(final_match_table, os.path.join(save_path, "table.pkl"))
    return final_match_table