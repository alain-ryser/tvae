from argparse import ArgumentParser
import torch
import torchvision
import pytorch_lightning as pl
from ..base_module import BaseModule
from ..models.encoder import AEEncoder
from ..models.decoder import ImageDecoder

from PIL import Image

class AutoEncoderExperiment(BaseModule):

    def __init__(
        self, 
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Modules for the forward function
        self.encoder = AEEncoder(pretrain=self.hparams.pretrain)
        self.decoder = ImageDecoder(pretrain=self.hparams.pretrain)

    def inference(self, x):
        _,_,H,W = x.shape
        with torch.no_grad():
            embedding = self.encoder(x)
            rec = self.decoder(embedding)
            rec = torch.relu(rec)
            return embedding, rec
        
    def forward(self, x):
        # Forward loop for predictions
        _,_,H,W = x.shape
        embedding = self.encoder(x)
        rec = self.decoder(embedding)
        rec = torch.relu(rec)
        return rec

    def training_step(self, batch, batch_idx):
        y = self.augment(batch[0])
        y_hat = self(y)
        loss = 1./y.shape[0] * self.rec_loss(y,y_hat)
        self.log('train_loss', loss,prog_bar=True)
        
        # Log first batch of images every couple of epochs
        if batch_idx == 0 and self.current_epoch % 64 == 0:
            grid = torchvision.utils.make_grid(torch.cat((y[:8,:,:,:],y_hat[:8,:,:,:]),dim=0))
            self.logger.experiment.add_image('train_recs',grid,self.current_epoch)
            
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch[0]
        y_hat = self(y)
        loss = 1./y.shape[0] * self.rec_loss(y,y_hat)
        self.log('val_loss', loss,prog_bar=False)
        if batch_idx == 0:
            grid = torchvision.utils.make_grid(torch.cat((y[:8,:,:,:],y_hat[:8,:,:,:]),dim=0))
            self.logger.experiment.add_image('val_recs',grid,self.current_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        y = batch[0]
        y_hat = self(y)
        loss = 1./y.shape[0] * self.rec_loss(y, y_hat)

        # log outputs
        self.log_dict({'test_loss': loss})
       