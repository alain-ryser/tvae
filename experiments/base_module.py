import torch
import pytorch_lightning as pl
from utils.transforms import Augment


class BaseModule(pl.LightningModule):

    def __init__(self,
                 pretrain=False,
                 augment=False,
                 **kwargs
                 ):
        super().__init__()
        
        # Save input arguments for checkpointing
        self.save_hyperparameters()
        
        # Set augmentation transform
        if augment:
            self.aug_transform = Augment()
        else:
            self.aug_transform = lambda x,y: x
        
        # Set default learning rate
        self.learning_rate = 0.001

        # Losses
        self.n_mc_samples = 1
        self.rec_loss = torch.nn.MSELoss(reduction='sum')


    def configure_optimizers(self):
        rec_optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return rec_optimizer
    
    def augment(self, x):
        with torch.no_grad():
            orig_shape = x.shape
            H,W = x.shape[-2:]
            T = None
            if len(x.shape)==5:
                T=x.shape[1]
            x = x.reshape(-1,1,H,W)
            x = self.aug_transform(x,T)
            x = x.reshape(*orig_shape)
            return x