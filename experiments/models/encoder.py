from utils import utilities
from torch import nn
import numpy as np
import pytorch_lightning as pl

def assemble_encoder(video=False):
    # Convolution
    layers = []
    if video:
        layers.append(Residual(25, 32, 3)) #64x64
    else:
        layers.append(Conv(1, 8, 3))
        layers.append(Residual(8, 32, 3)) #64x64
    layers.append(Residual(32, 64, 3)) #32x32
    layers.append(Residual(64, 64, 3)) #16x16
    layers.append(Residual(64, 128, 3)) #8x8
    layers.append(Residual(128, 256, 3)) #4x4
    layers.append(Residual(256, 512, 3)) #2x2
    layers.append(nn.AdaptiveAvgPool2d(1)) #1x1
    layers.append(nn.Flatten())

    return nn.Sequential(*layers)

class Conv(pl.LightningModule):

    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 padding='same',
                 stride=1,
                 batch_normalisation=True,
                 activation=True
                 ):
        super().__init__()

        layers = []
        layers.append(
            nn.Conv2d(in_ch, out_ch,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=(not batch_normalisation)
                      )
        )
        if batch_normalisation:
            layers.append(nn.BatchNorm2d(out_ch))
        if activation:
            layers.append(nn.ReLU())

        self.conv = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.conv(inputs)


class Residual(pl.LightningModule):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.pre_residual = Conv(in_ch, out_ch, kernel_size=kernel_size, stride=1)
        self.up_channel = Conv(in_ch, out_ch, kernel_size=1, stride=1)
        self.post_residual = Conv(out_ch, out_ch, kernel_size=kernel_size, stride=2, padding=1)

    def forward(self, x):
        conv_x = self.pre_residual(x)
        x = self.up_channel(x) + conv_x
        return self.post_residual(x)



class AEEncoder(pl.LightningModule):
    """
    Encoder for Stander AE architecture
    """

    def __init__(self, pretrain=False):
        super().__init__()
        
        # Encoder backbone
        self.encoder = assemble_encoder()
        
        # Extract z
        self.phi_z = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        if pretrain:
            utilities.load_pretrain(self)
    
    def forward(self, x):
        x = self.encoder(x)
        z = self.phi_z(x)
        return z
        
class VAEEncoder(pl.LightningModule):
    """
    VAE Encoder model
    """

    def __init__(self,pretrain=False):
        super().__init__()
        
        # Simple base encoder
        self.encoder = assemble_encoder()
        
        # Learn mean and log variance of latent space
        
        self.phi_z_mean = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.phi_z_log_var = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        if pretrain:
            utilities.load_pretrain(self)
    
    def forward(self, x):
        x = self.encoder(x)
        mu, log_var = self.phi_z_mean(x), self.phi_z_log_var(x)
        return mu, log_var
    
    
class TrajectoryEncoder(pl.LightningModule):
    """
    Latent trajectory model based variational encoder
    """

    def __init__(self, pretrain=False, trajectory_func='orig'):
        super().__init__()
        
        # Encoder of individual frames
        self.encoder = assemble_encoder(video=True)

        # Learn trajectory parameters
        self.phi_b = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.phi_f = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.phi_omega = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
        if trajectory_func == 'spiral':
            self.phi_velocity = nn.Sequential(
                nn.Linear(512, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        self.trajectory_func = trajectory_func
        
        if pretrain:
            utilities.load_pretrain(self)
        
    def forward(self, x):
        # Pass timeseries through encoder
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        feat = self.encoder(x)

        # Get traj params
        b = self.phi_b(feat)
        f = self.phi_f(feat)
        omega = 2*np.pi*self.phi_omega(feat)

        #Get velocity params if necessary
        if self.trajectory_func == 'spiral':
            velocity = self.phi_velocity(feat)
            return f, omega, velocity, b

        return f, omega, b

class TrajectoryVariationalEncoder(TrajectoryEncoder):
    """
    Latent trajectory model based variational encoder
    """

    def __init__(self, pretrain=False, trajectory_func='orig'):
        super().__init__(pretrain=False, trajectory_func=trajectory_func)
        self.phi_b_log_var =  nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Load pretrained weights
        if pretrain:
            utilities.load_pretrain(self)
        
    def forward(self, x):
        # Pass timeseries through encoder
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        feat = self.encoder(x)

        # Get traj params
        b_mean = self.phi_b(feat)
        b_log_var = self.phi_b_log_var(feat)
        f = self.phi_f(feat)
        omega = 2*np.pi*self.phi_omega(feat)

        #Get velocity params if necessary
        if self.trajectory_func == 'spiral':
            velocity = self.phi_velocity(feat)
            return f, omega, velocity, b_mean, b_log_var

        return f, omega, b_mean, b_log_var