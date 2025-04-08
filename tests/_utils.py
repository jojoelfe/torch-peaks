import torch
from torch_grid_utils import coordinate_grid
from torch_grid_utils.fftfreq_grid import dft_center

from torch_find_peaks.gaussians import Gaussian2D

def create_test_image(size:int=100, peaks: torch.tensor = torch.tensor([]), noise_level=0.1):
    """
    Create a test image with known Gaussian peaks.
    
    Args:
        size: Size of the square image
        peaks: (n,5) tensor of peak parameters (amplitude, x0, y0, sigma_x, sigma_y)
        noise_level: Level of noise to add
        
    Returns:
        Tuple containing:
        - Image tensor with Gaussian peaks
        - List of true peak parameters (amplitude, x0, y0, sigma_x, sigma_y)
    """
    # Create a blank image
    image = torch.randn((size, size)) * noise_level
        
    gaussian_model = Gaussian2D(
        amplitude=peaks[:, 0],
        center_x=peaks[:, 1],
        center_y=peaks[:, 2],
        sigma_x=peaks[:, 3],
        sigma_y=peaks[:, 4],
    )
    
    center = dft_center((size,size),rfft=False,fftshifted=True)
    grid = coordinate_grid((size,size),center=center)

    # Add Gaussian peaks to the image
    image += gaussian_model(grid).sum(dim=0)

    return image

