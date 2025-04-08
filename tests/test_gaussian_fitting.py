import torch
import numpy as np
import pytest
from torch_peaks.fit_gaussians import fit_gaussians_2d
from torch_peaks.find_peaks import peak_local_max_2d
from torch_peaks.gaussians import Gaussian2D



def test_fit_gaussians_2d_basic():
    """Test basic functionality of 2D Gaussian fitting."""
    # Create a test image with known peaks
    image, true_params = create_test_image(size=100, num_peaks=2, noise_level=0.05)
    
    # Detect peaks
    peak_coords = peak_local_max_2d(image, min_distance=10, threshold_abs=0.5)
    
    # Fit Gaussians to the peaks
    fitted_params, detected_peaks = fit_gaussians_2d(
        image,
        peak_coords=peak_coords,
        boxsize=20,
        max_iterations=500,
        learning_rate=0.01
    )
    
    # Check that we found the correct number of peaks
    assert len(fitted_params) == len(true_params)
    
    # Check that the fitted parameters are close to the true parameters
    for i, (fitted, true) in enumerate(zip(fitted_params, true_params)):
        # Find the closest true peak to this fitted peak
        distances = []
        for j, t in enumerate(true_params):
            dist = np.sqrt((fitted['x0'] - t['x0'])**2 + (fitted['y0'] - t['y0'])**2)
            distances.append((dist, j))
        
        # Sort by distance and get the closest true peak
        distances.sort()
        closest_idx = distances[0][1]
        closest_true = true_params[closest_idx]
        
        # Check that the parameters are close
        assert abs(fitted['amplitude'] - closest_true['amplitude']) < 0.5
        assert abs(fitted['x0'] - closest_true['x0']) < 2.0
        assert abs(fitted['y0'] - closest_true['y0']) < 2.0
        assert abs(fitted['sigma_x'] - closest_true['sigma_x']) < 1.0
        assert abs(fitted['sigma_y'] - closest_true['sigma_y']) < 1.0

def test_fit_gaussians_2d_no_peaks():
    """Test 2D Gaussian fitting when no peaks are detected."""
    # Create a blank image with no peaks
    image = torch.zeros((100, 100))
    
    # Try to fit Gaussians
    fitted_params, peak_coords = fit_gaussians_2d(
        image,
        min_distance=10,
        threshold_abs=1.0
    )
    
    # Check that no peaks were found
    assert len(fitted_params) == 0
    assert len(peak_coords) == 0

def test_fit_gaussians_2d_overlapping():
    """Test 2D Gaussian fitting with overlapping peaks."""
    # Create a test image with overlapping peaks
    image, true_params = create_test_image(size=100, num_peaks=3, noise_level=0.05)
    
    # Add an overlapping peak
    gaussian = Gaussian2D(
        amplitude=1.5,
        x0=50,
        y0=50,
        sigma_x=2.0,
        sigma_y=2.0
    )
    
    # Create coordinate grids
    y, x = torch.meshgrid(
        torch.arange(100),
        torch.arange(100),
        indexing='ij'
    )
    
    # Add the overlapping Gaussian to the image
    image += gaussian(x, y).detach()
    
    # Detect peaks
    peak_coords = peak_local_max_2d(image, min_distance=5, threshold_abs=0.5)
    
    # Fit Gaussians to the peaks
    fitted_params, detected_peaks = fit_gaussians_2d(
        image,
        peak_coords=peak_coords,
        boxsize=21,
        max_iterations=500,
        learning_rate=0.01
    )
    
    # Check that we found at least the original peaks plus the overlapping one
    assert len(fitted_params) >= len(true_params) + 1

def test_fit_gaussians_2d_with_provided_peaks():
    """Test 2D Gaussian fitting with manually provided peak coordinates."""
    # Create a test image with known peaks
    image, true_params = create_test_image(size=100, num_peaks=2, noise_level=0.05)
    
    # Manually specify peak coordinates (approximate positions of the true peaks)
    peak_coords = torch.tensor([
        [true_params[0]['y0'], true_params[0]['x0']],
        [true_params[1]['y0'], true_params[1]['x0']]
    ]).to(torch.int64)
    
    # Fit Gaussians to the peaks
    fitted_params, detected_peaks = fit_gaussians_2d(
        image,
        peak_coords=peak_coords,
        boxsize=21,
        max_iterations=500,
        learning_rate=0.01
    )
    
    # Check that we found the correct number of peaks
    assert len(fitted_params) == len(true_params)
    
    # Check that the fitted parameters are close to the true parameters
    for i, (fitted, true) in enumerate(zip(fitted_params, true_params)):
        assert abs(fitted['amplitude'] - true['amplitude']) < 0.5
        assert abs(fitted['x0'] - true['x0']) < 2.0
        assert abs(fitted['y0'] - true['y0']) < 2.0
        assert abs(fitted['sigma_x'] - true['sigma_x']) < 1.0
        assert abs(fitted['sigma_y'] - true['sigma_y']) < 1.0

def test_fit_gaussians_2d_different_boxsizes():
    """Test 2D Gaussian fitting with different box sizes."""
    # Create a test image with known peaks
    image, true_params = create_test_image(size=100, num_peaks=1, noise_level=0.05)
    
    # Detect peaks
    peak_coords = peak_local_max_2d(image, min_distance=10, threshold_abs=0.5)
    
    # Try different box sizes
    boxsizes = [11, 21, 31]
    results = []
    
    for boxsize in boxsizes:
        # Fit Gaussians to the peaks
        fitted_params, _ = fit_gaussians_2d(
            image,
            peak_coords=peak_coords,
            boxsize=boxsize,
            max_iterations=500,
            learning_rate=0.01
        )
        
        results.append(fitted_params[0])
    
    # Check that the results are similar regardless of box size
    for i in range(1, len(results)):
        assert abs(results[i]['amplitude'] - results[0]['amplitude']) < 0.2
        assert abs(results[i]['x0'] - results[0]['x0']) < 1.0
        assert abs(results[i]['y0'] - results[0]['y0']) < 1.0
        assert abs(results[i]['sigma_x'] - results[0]['sigma_x']) < 0.5
        assert abs(results[i]['sigma_y'] - results[0]['sigma_y']) < 0.5 