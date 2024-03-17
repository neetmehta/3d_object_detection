import torch
import torch.nn as nn
from importlib import import_module

class BaseTrainLoop:
    
    def __init__(self,
                 dataloader,
                 model,
                 loss,
                 optimizer_cfg,
                 no_of_epochs,
                 preprocess_func,
                 lr_schedular_cfg=None):
        
        self.dataloader = dataloader
        self.model = model
        self.loss = loss
        
        optimizer = getattr(import_module('torch.optim'), optimizer_cfg.pop('type'))
        if lr_schedular:
            lr_schedular = getattr(import_module('torch.optim.lr_scheduler'), lr_schedular_cfg.pop('type'))
        
        self.optimizer = optimizer(self.model.parameters(), **optimizer_cfg)
        lr_schedular = lr_schedular(self.optimizer, **lr_schedular_cfg)
        self.preprocess = preprocess_func
        self.num_epochs = no_of_epochs
        
    def run_single_epoch(self):
        
        for idx, batch_data in self.dataloader:
            batch_data = self.preprocess(batch_data)
            self.optimizer.zero_grad()
            pred = self.model(batch_data)
            loss = self.loss(pred, batch_data['target'])
            loss.backward()
            self.optimizer.step()
        
    def train(self):
        
        for _ in range(self.num_epochs):
            self.run_single_epoch()
            
            
def build_train_loop(cfg):
    
    