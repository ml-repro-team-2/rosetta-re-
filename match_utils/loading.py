import torch
import pickle
import os


def load_stats(root, device):
    '''Load table and stats.'''
    print("Loading...")
    file_name = os.path.join(root, "table.pkl")
    with open(file_name, 'rb') as f:
        table = pickle.load(f)
        table = table.to(device)
    
    with open(os.path.join(root,"discr_stats.pkl"), 'rb') as f:
        discr_stats = pickle.load(f)
        for key in discr_stats.keys():
          discr_stats[key] = (discr_stats[key][0].to(device),discr_stats[key][1].to(device))
                
        
    with open(os.path.join(root,"gan_stats.pkl"), 'rb') as f:
        gan_stats = pickle.load(f)
        for key in gan_stats.keys():
          gan_stats[key] = (gan_stats[key][0].to(device),gan_stats[key][1].to(device))
                
        
        
    print("Done")
    return table, gan_stats, discr_stats
