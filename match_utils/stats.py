import torch,torchvision
import tqdm

def get_mean_std(gan,gan_layers,discr,discr_layers,gan_mode,discr_mode,dataset,epochs,batch_size,device):
    '''Get the activation statistics from GAN and discr'''
    print("Collecting Dataset Statistics")

    with torch.no_grad():
        batch_gan_stats={}
        batch_discr_stats={}
        for layer in gan_layers:
            batch_gan_stats[layer]=[]
       
        for layer in discr_layers:
            batch_discr_stats[layer]=[]
            
        for iteration in tqdm.trange(0, epochs):
            z = dataset[0][iteration*batch_size: (iteration+1)*batch_size ].float()
            c = dataset[1][iteration*batch_size: (iteration+1)*batch_size ].float()
            if gan_mode=="biggan":
                img=gan(z,c,1)
                img=(img+1)/2

            
            for layer in gan_layers:   
                gan_activation=gan._retrieve_retained(layer,clear=True)#B,C,H,H
                                
                gan_activation=gan_activation.permute(1,0,2,3).contiguous()
                gan_activation=gan_activation.view(gan_activation.shape[0],-1)
                batch_gan_stats[layer].append((torch.mean(gan_activation,dim=-1,dtype=torch.float64).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device),\
                                    torch.std(gan_activation,dim=-1).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)))
                
                
            #clip condition here
            img = torch.nn.functional.interpolate(img, size = (224,224), mode = "bicubic")
            img = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))(img)
            _=discr(img)
            for layer in discr_layers:   
                discr_activation=discr._retrieve_retained(layer,clear=True)#B,C,H,H
                
                
                discr_activation=discr_activation.permute(1,0,2,3).contiguous()
                discr_activation=discr_activation.view(discr_activation.shape[0],-1)
                batch_discr_stats[layer].append((torch.mean(discr_activation,dim=-1,dtype=torch.float64).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device),\
                                    torch.std(discr_activation,dim=-1).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)))
                
        print("finished iterating for stats")
        
        final_gan_stats={}
        for layer in gan_layers:
            final_gan_stats[layer]=[0,0]
            for (mean,std) in batch_gan_stats[layer]:
                final_gan_stats[layer][0]+=mean
                final_gan_stats[layer][1]+=std**2
                
            final_gan_stats[layer][0]/=epochs
            final_gan_stats[layer][1]/=epochs
            final_gan_stats[layer][1]=torch.sqrt(final_gan_stats[layer][1])
            del batch_gan_stats[layer]
 
        final_discr_stats={}
        for layer in discr_layers:
            final_discr_stats[layer]=[0,0]
            for (mean,std) in batch_discr_stats[layer]:
                final_discr_stats[layer][0]+=mean
                final_discr_stats[layer][1]+=std**2
            final_discr_stats[layer][0]/=epochs
            final_discr_stats[layer][1]/=epochs
            final_discr_stats[layer][1]=torch.sqrt(final_discr_stats[layer][1])
            del batch_discr_stats[layer]
            torch.cuda.empty_cache()
            
        return final_gan_stats,final_discr_stats