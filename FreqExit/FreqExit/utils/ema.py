import torch
import torch.nn as nn
from copy import deepcopy
from collections import OrderedDict

class EMA:

    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
     
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        
     
        self.backup = {}

    def update(self):
     
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def store(self):
  
        
        self.backup = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()

    def copy_to_model(self):
      
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    def restore(self):
     
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        
       
        self.backup = {}

    def state_dict(self):
     
        return {
            'decay': self.decay,
            'shadow': deepcopy(self.shadow),  
            'backup': deepcopy(self.backup) if self.backup else {}  
        }

    def load_state_dict(self, state_dict):
     
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
        self.backup = state_dict['backup'] if 'backup' in state_dict else {}