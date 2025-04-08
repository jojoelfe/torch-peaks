import torch
from torch_grid_utils import coordinate_grid
from torch_find_peaks.find_peaks import find_peaks_2d
from torch_grid_utils.fftfreq_grid import dft_center

from _utils import create_test_image

def test_peak_picking_2d():
    peaks = torch.tensor([[1, 0, 0, 5, 5], [1, -20, -20, 2, 2]], dtype=torch.float32)
    data = create_test_image(size=100, peaks=peaks, noise_level=0.05)
    
    # Small distance and low threshold should pick extra peaks
    peak_detections = find_peaks_2d(data, min_distance=1, threshold_abs=0.3)
    assert peak_detections.shape[0] > 2
    # Appropriate values should find exactly the two peaks
    peak_detections = find_peaks_2d(data, min_distance=5, threshold_abs=0.5)
    assert peak_detections.shape[0] == 2
    assert torch.allclose(peak_detections[0], torch.tensor([30, 30]), atol=1.5)
    assert torch.allclose(peak_detections[1], torch.tensor([50, 50]), atol=1.5)

