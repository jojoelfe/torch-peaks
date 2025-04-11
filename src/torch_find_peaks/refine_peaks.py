from typing import Union, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch_grid_utils import coordinate_grid
from torch_grid_utils.fftfreq_grid import dft_center
from torch_subpixel_crop import subpixel_crop_2d, subpixel_crop_3d

from .gaussians import Gaussian2D, Gaussian3D


def _refine_peaks_2d_torch(
    image: torch.Tensor,
    peak_coords: torch.Tensor,
    boxsize: int,
    max_iterations: int,
    learning_rate: float,
    tolerance: float,
    amplitude: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
) -> torch.Tensor:
    """
    Internal function to refine the positions of peaks in a 2D tensor.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n, 5) containing the fitted parameters for each peak.
        Each row contains [amplitude, center_x, center_y, sigma_x, sigma_y].
    """
    # Ensure boxsize is even
    if boxsize % 2 != 0:
        raise ValueError("boxsize must be even")
    # Ensure shape of peak_coords
    if peak_coords.shape[1] != 2:
        raise ValueError("peak_coords must have shape (n, 2)")
    # Ensure peak_coords is on the same device as image
    if peak_coords.device != image.device:
        raise ValueError("peak_coords must be on the same device as image")
    num_peaks = peak_coords.shape[0]

    # Crop regions around peaks
    boxes = subpixel_crop_2d(image, peak_coords, boxsize).detach()
    # Prepare coordinates
    center = dft_center((boxsize, boxsize), rfft=False, fftshifted=True)
    grid = coordinate_grid((boxsize, boxsize), center=center, device=image.device)

    # Initialize model
    model = Gaussian2D(amplitude=amplitude,
                       center_x=torch.zeros_like(amplitude),
                       center_y=torch.zeros_like(amplitude),
                       sigma_x=sigma_x,
                       sigma_y=sigma_y).to(image.device)

    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Fit the Gaussians
    for _ in range(max_iterations):
        optimizer.zero_grad()

        # Calculate predicted values
        output = model(grid)
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

    # Combine the (...,1) model parameters to a (...,5) tensor
    # and add the peak coordinates
    fitted_params = torch.stack([
        model.amplitude,
        model.center_x + peak_coords[:, 1],
        model.center_y + peak_coords[:, 0],
        model.sigma_x,
        model.sigma_y
    ], dim=-1).reshape(num_peaks, 5)

    return fitted_params, loss.item()


def refine_peaks_2d(
    image: Any,
    peak_coords: Any,
    boxsize: int,
    max_iterations: int = 1000,
    learning_rate: float = 0.01,
    tolerance: float = 1e-6,
    amplitude: Union[torch.Tensor, float] = 1.,
    sigma_x: Union[torch.Tensor, float] = 1.,
    sigma_y: Union[torch.Tensor, float] = 1.,
) -> torch.Tensor:
    """
    Refine the positions of peaks in a 2D image by fitting 2D Gaussian functions.

    Parameters
    ----------
    image : Any
        A 2D tensor-like object (e.g., torch.Tensor, numpy.ndarray)
        containing the image data.
    peak_coords : Any
        A tensor-like object of shape (n, 2) containing the initial peak coordinates (y, x).
    boxsize : int
        Size of the region to crop around each peak (must be even).
    max_iterations : int, optional
        Maximum number of optimization iterations. Default is 1000.
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 0.01.
    tolerance : float, optional
        Convergence tolerance for the optimization. Default is 1e-6.
    amplitude : Union[torch.Tensor, float], optional
        Initial amplitude of the Gaussian. Default is 1.0.
    sigma_x : Union[torch.Tensor, float], optional
        Initial standard deviation in the x direction. Default is 1.0.
    sigma_y : Union[torch.Tensor, float], optional
        Initial standard deviation in the y direction. Default is 1.0.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n, 5) containing the fitted parameters for each peak.
        Each row contains [amplitude, center_x, center_y, sigma_x, sigma_y].
    """
    if not isinstance(image, torch.Tensor):
        image = torch.as_tensor(image)
    if not isinstance(peak_coords, torch.Tensor):
        peak_coords = torch.as_tensor(peak_coords)

    num_peaks = peak_coords.shape[0]
    if not isinstance(amplitude, torch.Tensor):
        amplitude = torch.tensor([amplitude] * num_peaks, device=image.device)
    if not isinstance(sigma_x, torch.Tensor):
        sigma_x = torch.tensor([sigma_x] * num_peaks, device=image.device)
    if not isinstance(sigma_y, torch.Tensor):
        sigma_y = torch.tensor([sigma_y] * num_peaks, device=image.device)

    return _refine_peaks_2d_torch(
        image=image,
        peak_coords=peak_coords,
        boxsize=boxsize,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        tolerance=tolerance,
        amplitude=amplitude,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
    )


def _refine_peaks_3d_torch(
    volume: torch.Tensor,
    peak_coords: torch.Tensor,
    boxsize: int,
    max_iterations: int,
    learning_rate: float,
    tolerance: float,
    amplitude: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
    sigma_z: torch.Tensor,
) -> torch.Tensor:
    """
    Internal function to refine the positions of peaks in a 3D tensor.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n, 7) containing the fitted parameters for each peak.
        Each row contains [amplitude, center_x, center_y, center_z, sigma_x, sigma_y, sigma_z].
    """
    # Ensure boxsize is even
    if boxsize % 2 != 0:
        raise ValueError("boxsize must be even")
    # Ensure shape of peak_coords
    if peak_coords.shape[1] != 3:
        raise ValueError("peak_coords must have shape (n, 3)")
    # Ensure peak_coords is on the same device as image
    if peak_coords.device != volume.device:
        raise ValueError("peak_coords must be on the same device as image")
    num_peaks = peak_coords.shape[0]

    # Crop regions around peaks
    boxes = subpixel_crop_3d(volume, peak_coords, boxsize).detach()

    # Prepare coordinates
    center = dft_center((boxsize, boxsize, boxsize), rfft=False, fftshifted=True)
    grid = coordinate_grid((boxsize, boxsize, boxsize), center=center, device=volume.device)

    # Initialize model
    model = Gaussian3D(amplitude=amplitude,
                       center_x=torch.zeros_like(amplitude),
                       center_y=torch.zeros_like(amplitude),
                       center_z=torch.zeros_like(amplitude),
                       sigma_x=sigma_x,
                       sigma_y=sigma_y,
                       sigma_z=sigma_z).to(volume.device)
    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    # Fit the Gaussians
    for _ in range(max_iterations):
        optimizer.zero_grad()

        # Calculate predicted values
        output = model(grid)
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
    # Combine the (...,1) model parameters to a (...,7) tensor
    # and add the peak coordinates
    fitted_params = torch.stack([
        model.amplitude,
        model.center_x + peak_coords[:, 2],
        model.center_y + peak_coords[:, 1],
        model.center_z + peak_coords[:, 0],
        model.sigma_x,
        model.sigma_y,
        model.sigma_z
    ], dim=-1).reshape(num_peaks, 7)
    return fitted_params, loss.item()


def refine_peaks_3d(
    volume: Any,
    peak_coords: Any,
    boxsize: int,
    max_iterations: int = 1000,
    learning_rate: float = 0.01,
    tolerance: float = 1e-6,
    amplitude: Union[torch.Tensor, float] = 1.,
    sigma_x: Union[torch.Tensor, float] = 1.,
    sigma_y: Union[torch.Tensor, float] = 1.,
    sigma_z: Union[torch.Tensor, float] = 1.,
) -> torch.Tensor:
    """
    Refine the positions of peaks in a 3D volume by fitting 3D Gaussian functions.

    Parameters
    ----------
    volume : Any
        A 3D tensor-like object (e.g., torch.Tensor, numpy.ndarray)
        containing the volume data.
    peak_coords : Any
        A tensor-like object of shape (n, 3) containing the initial peak coordinates (z, y, x).
    boxsize : int
        Size of the region to crop around each peak (must be even).
    max_iterations : int, optional
        Maximum number of optimization iterations. Default is 1000.
    learning_rate : float, optional
        Learning rate for the optimizer. Default is 0.01.
    tolerance : float, optional
        Convergence tolerance for the optimization. Default is 1e-6.
    amplitude : Union[torch.Tensor, float], optional
        Initial amplitude of the Gaussian. Default is 1.0.
    sigma_x : Union[torch.Tensor, float], optional
        Initial standard deviation in the x direction. Default is 1.0.
    sigma_y : Union[torch.Tensor, float], optional
        Initial standard deviation in the y direction. Default is 1.0.
    sigma_z : Union[torch.Tensor, float], optional
        Initial standard deviation in the z direction. Default is 1.0.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n, 7) containing the fitted parameters for each peak.
        Each row contains [amplitude, center_x, center_y, center_z, sigma_x, sigma_y, sigma_z].
    """
    if not isinstance(volume, torch.Tensor):
        volume = torch.as_tensor(volume)
    if not isinstance(peak_coords, torch.Tensor):
        peak_coords = torch.as_tensor(peak_coords)

    num_peaks = peak_coords.shape[0]
    if not isinstance(amplitude, torch.Tensor):
        amplitude = torch.tensor([amplitude] * num_peaks, device=volume.device)
    if not isinstance(sigma_x, torch.Tensor):
        sigma_x = torch.tensor([sigma_x] * num_peaks, device=volume.device)
    if not isinstance(sigma_y, torch.Tensor):
        sigma_y = torch.tensor([sigma_y] * num_peaks, device=volume.device)
    if not isinstance(sigma_z, torch.Tensor):
        sigma_z = torch.tensor([sigma_z] * num_peaks, device=volume.device)

    return _refine_peaks_3d_torch(
        volume=volume,
        peak_coords=peak_coords,
        boxsize=boxsize,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        tolerance=tolerance,
        amplitude=amplitude,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        sigma_z=sigma_z,
    )
