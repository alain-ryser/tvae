import torch
import numpy as np
import pytorch_lightning as pl
from utils.utilities import compute_kl_div, reparametrization_trick
from experiments.models.encoder import TrajectoryVariationalEncoder
from experiments.models.decoder import VariationalVideoDecoder
from ..base_module import BaseModule
from tqdm.auto import tqdm


class TrajectoryVAEExperiment(BaseModule):

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

        # Losses
        self.encoder = TrajectoryVariationalEncoder(pretrain=self.hparams.pretrain,trajectory_func=self.trajectory_func if self.trajectory_func!='circ' else 'orig')
        self.decoder = VariationalVideoDecoder(pretrain=self.hparams.pretrain,trajectory_func=self.trajectory_func if self.trajectory_func!='circ' else 'orig')

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

    def compute_elbo(self, x, x_hat, f, b_mus, b_log_vars, log=True):
        """
        Compute the elbo
        """

        # Get batch size, number of frames and latent dimensionality
        batch_size = x.shape[0]

        # Compute reconstruction loss
        reconstruction_loss = 0
        for mc_sample in x_hat:
            reconstruction_loss += self.rec_loss(mc_sample, x)
        reconstruction_loss *= 1. / (len(x_hat) * batch_size)

        # Regularize f to be between 60 and 120 bpm
        f_regu = 1/batch_size * (2-f).pow(10).sum()

        # Calculate b KL divergence with GMVAE
        b_kl = 1/batch_size * compute_kl_div(b_mus, b_log_vars).sum()

        # Log regularization term if loggin is switched on
        if log:
            self.log('b_kl', b_kl, prog_bar=False)
            self.log('train_rec', reconstruction_loss, prog_bar=False)
            self.log('f_mean', f.mean(), prog_bar=False)
            self.log('f_regu', f_regu, prog_bar=False)

        return reconstruction_loss + f_regu + b_kl

    def inference(self, sample, weight=None):
        B, T, C, H, W = sample.shape

        # Put tensor on GPU if available
        if self.device.type == 'cuda':
                sample=sample.cuda()
                
        if weight == None:
            with torch.no_grad():
                traj_params = list(self.encoder(sample))
                b_mean, b_log_var = traj_params[-2:]
                traj = self._evaluate_trajectory_function(*traj_params[:-2], b_mean, T)
                rec = self.decoder(traj)
                rec = rec.reshape(B, T, C, H, W)
                rec = torch.sigmoid(rec)
                return traj, rec
    
        # Prepare model
        self.freeze()
        self.eval()
        
        def total_variation_loss(vid, weight=1):
            B, T, H, W = vid.shape
            tv_h = torch.abs(vid[:,:,1:,:]-vid[:,:,:-1,:]).sum()
            tv_w = torch.abs(vid[:,:,:,1:]-vid[:,:,:,:-1]).sum()
            tv_t = torch.abs(vid[:,1:,:,:]-vid[:,:-1,:,:]).sum()
            return weight/B*(tv_h+tv_w+tv_t).sum()

        # Configure optimizer
        Y = torch.clone(sample).detach()
        with torch.no_grad():
            D = Y-self(Y)[0][0]
            
        D = torch.autograd.Variable(D, requires_grad=True)
        optimizer = torch.optim.Adam([D], 0.01)
        n_steps = 100
        # Optimize input
        for i in (pbar := tqdm(range(n_steps), leave=False)):
            args = self.encoder(Y-D)
            b_mean, b_log_vars = args[-2], args[-1]
            ELBO_X = 1./B * compute_kl_div(b_mean, b_log_vars).sum()
            # Compute TV regularizer
            TV_D = total_variation_loss(D.squeeze(),weight=weight)
            
            # Compute loss
            loss = ELBO_X + TV_D
            
            pbar.set_description(f"loss={loss.detach()}")
    
            # Optimizer step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        with torch.no_grad():
            traj_params = list(self.encoder(Y-D))
            b_mean, b_log_var = traj_params[-2:]
            traj = self._evaluate_trajectory_function(*traj_params[:-2], b_mean, T)
        return traj.detach(), (Y-D).detach()

    def forward(self, x):
        # Forward loop for predictions
        B, T, C, H, W = x.shape
        # Get trajectory function encoding
        args = self.encoder(x)
        b_mean, b_log_vars = args[-2], args[-1]
        recs = []
        for i in range(self.n_mc_samples):
            b = reparametrization_trick(b_mean, b_log_vars)
            emb =  self._evaluate_trajectory_function(*args[:-2],b,T)
            rec = self.decoder(emb)
            rec = rec.reshape(B, T, C, H, W)
            rec = torch.sigmoid(rec)
            recs.append(rec)
        return recs, args

    def training_step(self, batch, batch_idx):
        y = self.augment(batch[0])
        batch_size = y.shape[0]

        # Get reconstruction
        y_hat, args = self(y)
        f, b_mean, b_log_var = args[0],args[-2], args[-1]

        # Log first batch of videos every couple of epochs
        if batch_idx == 0 and self.current_epoch % 64 == 0:
            with torch.no_grad():
                video_recs = torch.cat((y[:8], y_hat[0][:8]), dim=-2)
                self.logger.experiment.add_video('train_recs', video_recs, global_step=self.current_epoch,
                                                 fps=12)
        # Compute ELBO
        loss = self.compute_elbo(
            y,
            y_hat,
            f,
            b_mean,
            b_log_var
        )

        self.log('train_loss', loss, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch[0]
        y_hat, _ = self(y)
        y_hat = y_hat[0]

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
        y_hat = y_hat[0]
        loss = 1. / (y.shape[0] * y.shape[1]) * self.rec_loss(y_hat, y)

        # Log first 8 videos of batch
        with torch.no_grad():
            video_recs = torch.cat((y[:8], y_hat[:8]), dim=-2)
            self.logger.experiment.add_video('test_recs', video_recs, batch_idx, fps=12)

        # log outputs
        self.log_dict({'test_loss': loss})
