import torch
import numpy as np
import pytest
from torch_find_peaks.refine_peaks import refine_peaks_2d, refine_peaks_3d
from _utils import create_test_image, create_test_volume



def test_refine_peaks_2d_basic():
    """Test basic functionality of 2D Gaussian fitting."""
    peaks = torch.tensor([[4, 50, 50, 5, 5], [1, 30, 30, 2, 2]], dtype=torch.float32)
    data = create_test_image(size=100, peaks=peaks, noise_level=0.05)
   
    # Fit Gaussians to the peaks
    fitted_params,_ = refine_peaks_2d(
        data,
        peak_coords=torch.tensor([[45, 49],[28, 32]], dtype=torch.float32),
        boxsize=20,
        max_iterations=500,
        learning_rate=0.05,
        tolerance=1e-8,
    )
    print(fitted_params)
    # Check that we found the correct number of peaks
    assert len(fitted_params) == len(peaks)
    
    assert torch.allclose(fitted_params[0], peaks[0], atol=1e-1)
    assert torch.allclose(fitted_params[1], peaks[1], atol=1e-1)

def test_refine_peaks_3d_basic():
    """Test basic functionality of 3D Gaussian fitting."""
    peaks = torch.tensor([[4, 50, 50, 50, 5, 5, 5], [1, 30, 30, 30, 2, 2, 2]], dtype=torch.float32)
    data = create_test_volume(size=100, peaks=peaks, noise_level=0.05)
    
    # Fit Gaussians to the peaks
    fitted_params,_ = refine_peaks_3d(
        data,
        peak_coords=torch.tensor([[45, 49,53],[28, 32,30]], dtype=torch.float32),
        boxsize=20,
        max_iterations=500,
        learning_rate=0.05,
        tolerance=1e-8,
    )
    
    # Check that we found the correct number of peaks
    assert len(fitted_params) == len(peaks)
    
    assert torch.allclose(fitted_params[0], peaks[0], atol=1e-1)
    assert torch.allclose(fitted_params[1], peaks[1], atol=1e-1)