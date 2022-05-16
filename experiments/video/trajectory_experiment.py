import torch
import pytorch_lightning as pl
import numpy as np
from utils.utilities import compute_kl_div, reparametrization_trick
from ..base_module import BaseModule
from experiments.models.encoder import TrajectoryEncoder
from experiments.models.decoder import VideoDecoder


class TrajectoryExperiment(BaseModule):

    def __init__(self,
                 trajectory_func='orig',
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.save_hyperparameters('trajectory_func')

        # Modules for the forward function
        self.trajectory_func = self.hparams.trajectory_func
        if self.trajectory_func not in ['orig','spiral','circ']:
            raise ValueError(f"trajectory function should be either 'orig' or 'spiral', not {self.trajectory_func}")
        self.encoder = TrajectoryEncoder(pretrain=self.hparams.pretrain,trajectory_func=self.trajectory_func if self.trajectory_func!='circ' else 'orig')
        self.decoder = VideoDecoder(pretrain=self.hparams.pretrain,trajectory_func=self.trajectory_func if self.trajectory_func!='circ' else 'orig')

    def _evaluate_trajectory_function(self, *args):
        if self.trajectory_func == 'orig':
            return self._evaluate_orig_trajectory_function(*args)
        elif self.trajectory_func == 'spiral':
            return self._evaluate_spiral_trajectory_function(*args)
        elif self.trajectory_func =='circ':
            return self._evaluate_circ_trajectory_function(*args)
        else:
            raise ValueError("Argument trajectory_func should be either 'orig' or 'spiral'.")

    def _evaluate_spiral_trajectory_function(self, f, omega, v, b, T):
        """
        Assemble timeseries from latent trajectory model
        """
        # Get batch size and dimension of latent space
        B, L = b.shape

        # Span timesteps
        half_frame_rate = 12
        t_inp = torch.arange(T).type_as(b).repeat(B, 1) / half_frame_rate

        # Prepare trajectory params for broadcasting
        t_inp = t_inp.reshape(B,T,1)
        f = f.reshape(B,1,1)
        v = v.reshape(B,1,1).repeat(1,1,L)
        v[:,:,1] = -v[:,:,1]
        omega = omega.reshape(B,1,1)

        # Evaluate latent trajectory at each timestep
        state = 2 * np.pi * f * t_inp - omega

        # Assemble trajectory
        trajectory = torch.zeros(B, T, L).type_as(b)
        trajectory[:,:,0] = (torch.cos(state)-torch.sin(state)).squeeze()
        trajectory[:,:,1:] = torch.cos(state)+torch.sin(state)
        trajectory = trajectory+b.unsqueeze(1)+t_inp*v

        return trajectory
    
    def _evaluate_orig_trajectory_function(self, f, omega, b, T):
        """
        Assemble timeseries from latent trajectory model
        """
        # Get batch size and dimension of latent space
        B, L = b.shape

        # Span timesteps
        half_frame_rate = 12
        t_inp = torch.arange(T).type_as(b).repeat(B, 1) / half_frame_rate

        # Prepare trajectory params for broadcasting
        t_inp = t_inp.reshape(B,T,1)
        f = f.reshape(B,1,1)
        omega = omega.reshape(B,1,1)

        # Evaluate latent trajectory at each timestep
        state = 2 * np.pi * f * t_inp - omega

        # Assemble trajectory
        trajectory = torch.zeros(B, T, L).type_as(b)
        trajectory[:,:,0] = (torch.cos(state)-torch.sin(state)).squeeze()
        trajectory[:,:,1:] = torch.cos(state)+torch.sin(state)
        trajectory = trajectory+b.unsqueeze(1)

        return trajectory
    
    def _evaluate_circ_trajectory_function(self, f, omega, b, T):
        """
        Assemble timeseries from latent trajectory model
        """
         # Get batch size and dimension of latent space
        B, L = b.shape

        # Span timesteps
        half_frame_rate = 12
        t_inp = torch.arange(T).type_as(b).repeat(B, 1) / half_frame_rate

        # Prepare trajectory params for broadcasting
        t_inp = t_inp.reshape(B,T,1)
        f = f.reshape(B,1,1)
        omega = omega.reshape(B,1,1)

        # Evaluate latent trajectory at each timestep
        state = 2 * np.pi * f * t_inp - omega

        # Assemble trajectory
        trajectory = torch.zeros(B, T, L).type_as(b)
        trajectory[:,:,0] = torch.cos(state).squeeze()
        trajectory[:,:,1] = torch.sin(state).squeeze()
        trajectory = trajectory+b.unsqueeze(1)

        return trajectory


    def inference(self, sample):       
        B, T, C, H, W = sample.shape

        # Put tensor on GPU if available
        if self.device.type == 'cuda':
            sample=sample.cuda()
                
        rec, args = self(sample)
        return self._evaluate_trajectory_function(*args, T), rec

    def forward(self, x):
        # Forward loop for predictions
        B, T, C, H, W = x.shape
        args = self.encoder(x)
        emb = self._evaluate_trajectory_function(*args, T)
        rec = self.decoder(emb)
        rec = rec.reshape(B, T, C, H, W)
        rec = torch.sigmoid(rec)
        return rec, args

    def training_step(self, batch, batch_idx):
        
        # Apply data augmentation if needed
        y = self.augment(batch[0])
        batch_size = y.shape[0]

        # Get reconstruction        
        y_hat, args  = self(y)
        
        f,r = args[0],args[2]
        
        self.log('f_mean', f.mean(), prog_bar=False)

        # Log first batch of videos every couple of epochs
        if batch_idx == 0 and self.current_epoch % 64 == 0:
            with torch.no_grad():
                video_recs = torch.cat((y[:8], y_hat[:8]), dim=-2)
                self.logger.experiment.add_video('train_recs', video_recs, global_step=self.current_epoch,
                                                 fps=12)
        # Compute loss
        loss = 1/batch_size * (
            self.rec_loss(y_hat, y.detach()) + # reconstruction error
            (2-f).pow(10).sum() # reasonable f
        )

        self.log('train_loss', loss, prog_bar=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch[0]
        y_hat, _ = self(y)

        # Compute loss
        loss = 1. / (y.shape[0] * y.shape[1]) * self.rec_loss(y_hat, y)

        self.log('val_loss', loss, prog_bar=False, sync_dist=True)

        if batch_idx == 0:
            with torch.no_grad():
                video_recs = torch.cat((y, y_hat), dim=-2)
                self.logger.experiment.add_video('val_recs', video_recs, global_step=self.current_epoch, fps=12)
        return loss

    def test_step(self, batch, batch_idx):
        y = batch[0]
        y_hat, _ = self(y)
        loss = 1. / (y.shape[0] * y.shape[1]) * self.rec_loss(y_hat, y)

        # Log first 8 videos of batch
        with torch.no_grad():
            video_recs = torch.cat((y[:8], y_hat[:8]), dim=-2)
            self.logger.experiment.add_video('test_recs', video_recs, batch_idx, fps=12)

        # log outputs
        self.log_dict({'test_loss': loss})
