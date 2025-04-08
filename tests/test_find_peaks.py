import torch
from torch_grid_utils import coordinate_grid
from torch_find_peaks.find_peaks import find_peaks_2d
from torch_grid_utils.fftfreq_grid import dft_center

from _utils import create_test_image

def test_peak_picking():

    data = create_test_image(size=100, peaks=torch.tensor([[1, 50, 50, 2, 2], [1, 30, 30, 2, 2]]), noise_level=0.05)
    
    peak_detections = find_peaks_2d(data, min_distance=1, threshold_abs=5)
    assert peak_detections.shape[0] > 1
    peak_detections = find_peaks_2d(data, min_distance=5, threshold_abs=5)
    assert peak_detections.shape[0] == 1
    assert torch.max(peak_detections[0] - torch.tensor([34, 54])) < 2

