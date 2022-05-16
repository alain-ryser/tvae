import torch
import torchvision
from tqdm.auto import tqdm
import pytorch_lightning as pl
from utils.utilities import compute_kl_div, reparametrization_trick
from ..models.encoder import VAEEncoder
from ..models.decoder import VariationalImageDecoder
from ..base_module import BaseModule

class VAEExperiment(BaseModule):

    def __init__(
        self, 
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Modules for the forward function
        self.encoder = VAEEncoder(pretrain=self.hparams.pretrain)
        self.decoder = VariationalImageDecoder(pretrain=self.hparams.pretrain)
    
    def compute_elbo(self, x, x_hat, mus, log_vars, kl_div_weight=1, log = True):
        """
        Compute the elbo
        """
        batch_size = x.shape[0]
        reconstruction_loss = 0 
        for mc_sample in x_hat:
            reconstruction_loss += self.rec_loss(mc_sample,x)
        reconstruction_loss *= 1./len(x_hat)
        kl_div = compute_kl_div(mus, log_vars).sum()
        if log:
            self.log('train_rec', (1./batch_size)*reconstruction_loss,prog_bar=False)
            self.log('train_kl_div', (1./batch_size)*kl_div_weight*kl_div,prog_bar=False)
        return (1./batch_size)*(reconstruction_loss + kl_div_weight*kl_div)
    
    def inference(self, y, weight=None):
        B, C, H, W = y.shape
        # Return forward pass 
        if weight == None:
            with torch.no_grad():
                y_hat, means, log_vars = self(y)
                return reparametrization_trick(means, log_vars), y_hat[0]
        
        # Prepare model
        self.freeze()
        self.eval()
        
        def total_variation_loss(img, weight=1):
            B, T, H, W = img.shape
            tv_h = torch.abs(img[:,:,1:,:]-img[:,:,:-1,:]).sum()
            tv_w = torch.abs(img[:,:,:,1:]-img[:,:,:,:-1]).sum()
            return weight/B*(tv_h+tv_w).sum()

        # Configure optimizer
        Y = torch.clone(y).detach()
        with torch.no_grad():
            D = Y-self(Y)[0][0]
        D = torch.autograd.Variable(D, requires_grad=True)
        optimizer = torch.optim.Adam([D], 0.01)
        n_steps = 100
        # Optimize input
        for i in (pbar := tqdm(range(n_steps), leave=False)):
            means, log_vars = self.encoder(Y-D)
            ELBO_X = 1./B * compute_kl_div(means, log_vars).sum()
            # Compute TV regularizer
            TV_D = total_variation_loss(D,weight=weight)
            
            # Compute loss
            loss = ELBO_X + TV_D
            
            pbar.set_description(f"loss={loss.detach()}")
    
            # Optimizer step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            y_hat, means, log_vars = self(Y-D)
            emb = reparametrization_trick(means, log_vars)
        return emb.detach(), (Y-D).detach()
    
    
    def forward(self, x):
        # Forward loop for predictions
        _,_,H,W = x.shape
        means, log_vars = self.encoder(x)
        recs = []
        for i in range(self.n_mc_samples):
            embedding = reparametrization_trick(means, log_vars)
            rec = self.decoder(embedding)
            rec = torch.relu(rec)
            recs.append(rec)
        return recs, means, log_vars
    
    def training_step(self, batch, batch_idx):
        y = self.augment(batch[0])
        batch_size = y.shape[0]
        y_tilde = add_speckle_noise(y)
        
        # Get reconstruction
        y_hat, means, log_vars = self(y_tilde)
        
        # Log first batch of images every couple of epochs
        if batch_idx == 0 and self.current_epoch % 64 == 0:
            grid = torchvision.utils.make_grid(torch.cat((y[:8,:,:,:],y_hat[0][:8,:,:,:]),dim=0))
            self.logger.experiment.add_image('train_recs',grid,self.current_epoch)
        
        # Compute loss
        loss = self.compute_elbo(y, y_hat, means, log_vars)
        
        self.log('train_loss', loss,prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch[0]
        y_hat, _, _ = self(y)
        y_hat = y_hat[0]
        
        # Compute loss
        loss = 1./y.shape[0] * self.rec_loss(y_hat,y)
        
        self.log('val_loss', loss,prog_bar=False)
        if batch_idx == 0:
            grid = torchvision.utils.make_grid(torch.cat((y[:8,:,:,:],y_hat[:8,:,:,:]),dim=0))
            self.logger.experiment.add_image('val_recs',grid,self.current_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        y = batch[0]
        y_hat, _, _ = self(y)
        loss = 0 
        for mc_sample in y_hat:
            loss += self.rec_loss(mc_sample,y)
        loss *= 1./(len(y_hat)*y.shape[0])
        
        # Log first 8 images of batch
        grid = torchvision.utils.make_grid(torch.cat((y[:4,:,:,:],y_hat[0][:4,:,:,:]),dim=0))
        self.log_dict({'test_loss': loss})
