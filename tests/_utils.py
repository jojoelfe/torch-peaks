import torch
from torch_grid_utils import coordinate_grid

from torch_find_peaks.gaussians import Gaussian2D, Gaussian3D


def create_test_image(size:int=100, peaks: torch.tensor = torch.tensor([]), noise_level=0.1):
    """
    Create a test image with known Gaussian peaks.
    
    Args:
        size: Size of the square image
        peaks: (n,5) tensor of peak parameters (amplitude, x0, y0, sigma_x, sigma_y)
        noise_level: Level of noise to add
        
    Returns:
        - Image tensor with Gaussian peaks
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

    grid = coordinate_grid((size,size))

    # Add Gaussian peaks to the image
    image += gaussian_model(grid).sum(dim=0)

    return image

def create_test_volume(size:int=100, peaks: torch.tensor = torch.tensor([]), noise_level=0.1):
    """
    Create a test volume with known Gaussian peaks.
    
    Args:
        size: Size of the square image
        peaks: (n,7) tensor of peak parameters (amplitude, center_x, center_y, center_z, sigma_x, sigma_y, sigma_z)
        noise_level: Level of noise to add
        
    Returns:
        - Image tensor with Gaussian peaks
    """
    # Create a blank image
    image = torch.randn((size, size, size)) * noise_level

    gaussian_model = Gaussian3D(
        amplitude=peaks[:, 0],
        center_x=peaks[:, 1],
        center_y=peaks[:, 2],
        center_z=peaks[:, 3],
        sigma_x=peaks[:, 4],
        sigma_y=peaks[:, 5],
        sigma_z=peaks[:, 6],
    )

    grid = coordinate_grid((size,size,size))

    # Add Gaussian peaks to the image
    image += gaussian_model(grid).sum(dim=0)

    return image
