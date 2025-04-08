import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Union, List, Dict

from torch_subpixel_crop import subpixel_crop_3d, subpixel_crop_2d
from torch_grid_utils import coordinate_grid
from torch_grid_utils.fftfreq_grid import dft_center

from .gaussians import Gaussian2DList, Gaussian3DList



def fit_gaussians_2d(
    image: torch.Tensor,
    peak_coords: torch.Tensor,
    boxsize: int,
    max_iterations: int = 1000,
    learning_rate: float = 0.01,
    tolerance: float = 1e-6,
    amplitude: float = 1.,
    x0: float = 0.,
    y0: float = 0.,
    sigma_x: float = 1.,
    sigma_y: float = 1.,
) -> Tuple[List[Dict[str, float]], torch.Tensor]:
    """
    Fit 2D Gaussians to peaks in an image.
    
    Args:
        image: 2D tensor containing the image data
        peak_coords: Tensor of peak coordinates (y, x). 
        min_distance: Minimum distance between peaks (used only if peak_coords is None)
        threshold_abs: Absolute threshold for peak detection (used only if peak_coords is None)
        exclude_border: Number of pixels to exclude from the border (used only if peak_coords is None)
        boxsize: Size of the region to crop around each peak (must be odd)
        max_iterations: Maximum number of optimization iterations
        learning_rate: Learning rate for the optimizer
        tolerance: Convergence tolerance
        
    Returns:
        Tuple containing:
        - List of dictionaries with fitted parameters
        - Tensor of peak coordinates
    """

    
    # Ensure boxsize is even
    if boxsize % 2 != 0:
        raise ValueError("boxsize must be even")
    
    # Crop regions around peaks
    boxes = subpixel_crop_2d(image, peak_coords, boxsize)
    # Prepare coordinates
    center = dft_center((boxsize,boxsize),rfft=False,fftshifted=True)
    grid = coordinate_grid((boxsize,boxsize),center=center) 
    y_coords, x_coords = grid[::,0], grid[::,1]
    
    # Initialize model
    num_gaussians = len(peak_coords)
    model = Gaussian2DList(num_gaussians, amplitude=amplitude, x0=x0, y0=y0, sigma_x=sigma_x, sigma_y=sigma_y).to(image.device)
    
    
    
    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Fit the Gaussians
    for _ in range(max_iterations):
        optimizer.zero_grad()
        
        # Calculate predicted values
        output = model(x_coords, y_coords, torch.zeros_like(x_coords)).reshape(num_gaussians, boxsize, boxsize)
        
        # Calculate loss
        loss = criterion(output, boxes)
        
        # Check convergence
        if loss.item() < tolerance:
            break
        
        # Backpropagate and update
        loss.backward()
        optimizer.step()
        
        # Ensure positive values for amplitude and sigma
        with torch.no_grad():
            model.amplitude.data.clamp_(min=0)
            model.sigma_x.data.clamp_(min=0.001)
            model.sigma_y.data.clamp_(min=0.001)
            model.sigma_z.data.clamp_(min=0.001)
    
    # Extract fitted parameters
    fitted_params = []
    for i in range(num_gaussians):
        fitted_params.append({
            'amplitude': model.amplitude[i].item(),
            'x0': model.x0[i].item(),
            'y0': model.y0[i].item(),
            'sigma_x': model.sigma_x[i].item(),
            'sigma_y': model.sigma_y[i].item(),
            'loss': loss.item()
        })
    
    return fitted_params
