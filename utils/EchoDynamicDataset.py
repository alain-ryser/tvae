import torch
import echonet
from torch.utils import data
from torchvision import transforms
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import os

class EchoDynamicDataset(data.Dataset):
    """
    Dataset for echo-dynamic preprocessing
    """
    def __init__(self, data_dir, frames=25, resize=128, split='train', num_worker=4, cache_dir='.'):

        # Define data transforms that should be applied to dataset
        self.size = resize
        self.num_frames = frames
        
        # Load data if in cache, else generate cache
        cache_fn = f"echo_dynamic_vids_split{split}_len{frames}.pt"
        if not Path(os.path.join(cache_dir,cache_fn)).exists():
            dynamic_dataset = echonet.datasets.Echo(root=data_dir, split=split, length=frames, period=4)
            self.rebuild_video_cache(cache_fn, cache_dir, dynamic_dataset,procs=num_worker)
        self.data = torch.load(os.path.join(cache_dir,cache_fn))
    
    def preprocess(self, sample):
        """
        Preprocess video frames
        """
        sample = sample[0]
        # Get correct shape
        sample = np.swapaxes(sample,1,0)
        # Collapse channels
        sample = np.median(sample, axis=1, keepdims=True)
        # Assemble tensor
        sample = torch.from_numpy(sample.astype(np.ubyte))
        # Resize correctly
        sample = transforms.functional.resize(sample, (self.size,self.size))
        # Histogram equalization
        background_idx = sample==0
        sample[background_idx]= torch.randint_like(sample,sample[sample!=0].min(),255)[background_idx]
        sample = transforms.functional.equalize(sample)
        sample[background_idx] = 0
        # Normalization
        sample = sample.float()/255.
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return transforms.functional.resize(sample, (self.size,self.size)), 0, 'dummy_patient_id'
    
    

    def rebuild_video_cache(self,fn, cache_dir, dataset, procs):
        Path(cache_dir).mkdir(exist_ok=True)
        data = []
        
        for sample in tqdm(dataset):
            preprocessed_sample = self.preprocess(sample)
            if self.num_frames is not None:
                data.append(preprocessed_sample)
            else:
                data += [x for x in preprocessed_sample]
                    

        # Save data
        torch.save(data,os.path.join(cache_dir,fn))