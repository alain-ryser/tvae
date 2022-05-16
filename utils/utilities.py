import torch
import os
from pathlib import Path

def reparametrization_trick(means, log_vars):
    dist = torch.distributions.normal.Normal(means,torch.exp(log_vars/2))
    return dist.rsample()

def compute_kl_div(mus, log_vars, other_mus = torch.tensor(0), other_log_vars=torch.tensor(0)):
    """
    Compute KL Divergence between two Gaussians
    """
    kl_div = 0.5*torch.sum(1./torch.exp(other_log_vars) * torch.exp(log_vars) + (other_mus-mus)*1./torch.exp(other_log_vars)*(other_mus-mus) - 1. + other_log_vars - log_vars,dim=1)
    return kl_div

def load_pretrain(model):
    """
    Load weights pretrained on EchoDynamic
    """
    # Weight tensors
    curr_dir = Path(__file__).parent.resolve()
    pretrain_dir = Path(os.path.join(curr_dir, '..','pretrain')).resolve()
    weight_path = os.path.join(pretrain_dir, f'{model.__class__.__name__}{"_"+model.trajectory_func if hasattr(model, "trajectory_func") else ""}.pt')
    weights = torch.load(weight_path, map_location=model.device)

    # Load weights form state dict
    missing, unexpected = model.load_state_dict(weights,strict=False)
    if missing != []:
        print(f"Couldn't load full model from state dict, missing keys: {', '.join(missing)}")
    if unexpected != []:
        print(f"Found unexpected keys in state dict: {', '.join(unexpected)}")