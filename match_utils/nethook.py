import torch,types,copy
from collections import OrderedDict, defaultdict
#from einops import rearrange
import numpy as np

class InstrumentedModel(torch.nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model
        self._retained=OrderedDict()
        self._detach_retained={}
        # self._editargs = defaultdict(dict)
        # self._editrule = {}
        self._hooked_layer=[]
        self._old_forward={}
        # if isinstance(model, torch.nn.Sequential):
        #     self._hook_sequential()
    
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        self.close()
    
    def forward(self,*inputs,**kwargs):
        '''forward it through the original model'''
        return self.model(*inputs,**kwargs)
    
    def retain_layers(self,layernames,detach=True):
        '''retain a list of layers'''
        #to reatin a single layer just call retain_layer([layername])
        self._add_hooks(layernames)
        for layername in layernames:
            if layername not in self._retained:
                self._retained[layername]=None
                self._detach_retained[layername]=detach

    def stop_retaining_layers(self,layernames):
        '''stop retaining a set of retained layers'''
        #check
        for layername in layernames:
            if layername in self._retained:
                del self._retained[layername]
                del self._detach_retained[layername]
                self._unhook_layer(layername)
            else:
                raise ValueError(f"{layername} has not been retained")


    def _add_hooks(self,layernames):
        '''set up layers to be hooked'''
        needed_to_hook=[]
        for layername in layernames:
            if layername not in self._hooked_layer:
                needed_to_hook.append(layername)
        if not needed_to_hook:
            return
        for layername,layer in self.model.named_modules():
            if layername in needed_to_hook:
                self._hook_layer(layer,layername)
                needed_to_hook.remove(layername)
        if needed_to_hook:
            raise ValueError(f'layer {layername} not found in model')
        
    def _hook_layer(self,layer,layername):
        '''
        this modifies the forward of the layer to retain the forward pass layer activations
        '''
        if layername in self._hooked_layer:
            raise ValueError(f'layes {layername} already hooked')
        # if layername in self._old_forward:#?
        #     raise ValueError('Layer %s already hooked' % layername)
        self._hooked_layer.append(layername)
        self._old_forward[layername]=(layer,layer.forward)
        original_forward=layer.forward
        editor=self
        def new_forward(self,*inputs,**kwargs):
            og_x=original_forward(*inputs,**kwargs)
            x=editor._postprocess_forward(og_x,layername)
            return x
        layer.forward=types.MethodType(new_forward,layer)

    def _unhook_layer(self,layername):
        if layername in self._retained:
            del self._retained[layername]
            del self._detach_retained[layername]
        layer,old_forward=self._old_forward[layername]
        #check
        layer.forward=old_forward
        del self.old_forward[layername]
        self._hooked_layer.remove(layername)

    def _postprocess_forward(self,x,layername):
        '''process is run during forward pass to retain activation'''
        if layername in self._retained:
            if self._detach_retained[layername]:
                self._retained[layername]=x.detach()
            else:
                self._retained[layername]=x
            return x
        
    def _retrieve_retained(self,layername=None,clear=False):
        if layername is None:
            #default give the first retained
            layername =self._hooked_layer[0]
        result=self._retained[layername]
        if clear:
            self._retained[layername]=None
        return result