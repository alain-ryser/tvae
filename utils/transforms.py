from torchvision import transforms
import torch
import math
import random
import torchvision.transforms.functional as F

class GaussianSmoothing():
    """
    Do Gaussian filtering with low sigma to smooth out edges
    """
    def __init__(self, kernel_size=11, sigma=0.5):
        # Define Gaussian Kernel
        self.gaussian_kernel = self._generate_gaussian_kernel(kernel_size =kernel_size, sigma = sigma)
                                                              
    def _generate_gaussian_kernel(self, kernel_size, sigma):
        """
        Code from https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
        """
        
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) /\
                              (2*variance)
                          )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        
        # Set pytorch convolution from gaussian kernel
        gaussian = torch.nn.Conv2d(1, 1, kernel_size,padding=kernel_size//2,padding_mode='zeros', bias=False)
        gaussian.weight.requires_grad = False
        gaussian.weight[:,:] = gaussian_kernel
        return gaussian
    
    def __call__(self, sample):
        sample, p_id = sample
        # Slightly smooth out edges with Gaussian Kernel
        with torch.no_grad():
            if len(sample.shape) == 3:
                sample = self.gaussian_kernel(sample.unsqueeze(0)).squeeze(0)
            else:
                sample = self.gaussian_kernel(sample)
        return sample, p_id

class Augment():
    """
    Randomly perform a range of data augmentation
    including translation, rotation, scaling, salt & pepper noise,
    brightness adjustment, Gamma Correction, blurring and sharpening the image
    """
    def __init__(self):

        # Define augmentation transforms
        self.intensity_transformations = [
            RandomSharpness(),
            RandomBrightnessAdjustment(),
            SaltPepperNoise()
        ]
        shear = 3
        self.positional_transformations = transforms.RandomAffine(10,translate=(0.1,0.1), scale=(0.75,1.25),shear=(shear,shear,shear,shear),fill=0)
    
    def select_samples(self,sample, T, prob=0.5):
        B = sample.shape[0]
        prob = 1-prob
        if T is None:
            return prob<torch.rand(B)
        else:
            B = B//T
            idx = prob<torch.rand(B)
            idx = idx.unsqueeze(0).repeat(T,1).T.flatten()
            return idx

    def __call__(self,sample, T=None):
        # Apply intensity transformations
        B = sample.shape[0]

        # Apply positional transformations a couple of times with low probability
        for i in range(10):
            selected = self.select_samples(sample, T, prob=0.1)
            if torch.any(selected):
                sample[selected] = self.positional_transformations(sample[selected])
        for t in self.intensity_transformations:
            selected = self.select_samples(sample, T, prob=0.3)
            if torch.any(selected):
                sample[selected] = t(sample[selected])

        return sample

class SaltPepperNoise():
    """
    Add Salt and Pepper noise on top of the image
    """
    def __init__(self, thresh=0.005):
        self.thresh = thresh

    def __call__(self, sample):
        noise = torch.rand(sample.shape)
        sample[noise < self.thresh] = 0
        sample[noise > 1-self.thresh] = 1
        return sample

class RandomBrightnessAdjustment():
    """
    Randomly adjust brightness
    """
    def __call__(self, sample):
        rand_factor = random.random()*0.7 + 0.5
        sample = F.adjust_brightness(sample, brightness_factor=rand_factor)
        return sample

class RandomSharpness():
    """
    Randomly increase oder decrease image sharpness
    """
    def __call__(self, sample):
        if 0.5 < torch.rand(1):
            rand_factor = random.random()*7 + 1
        else:
            rand_factor = random.random()
        sample = F.adjust_sharpness(sample, sharpness_factor=rand_factor)
        return sample

def get_transforms(
        resize=256,
        augment=True,
        with_pid=False,
        dataset_orig_img_scale=0.25
):
    """
    Compose a set of prespecified transformation using the torchvision transform compose class
    """
    return transforms.Compose(
        [
            Augment(orig_img_scale=dataset_orig_img_scale, size=resize, return_pid=with_pid) if augment else Identity(),
        ]
    )
