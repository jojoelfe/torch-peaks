import torch
from torch_grid_utils import coordinate_grid
from torch_peaks.find_peaks import peak_local_max_2d
from torch_grid_utils.fftfreq_grid import dft_center

def test_peak_picking():

    box_length = 100
    center = dft_center((box_length,box_length),rfft=False,fftshifted=True)
    grid = coordinate_grid((box_length,box_length),center=center) * 0.1
    noise = torch.randn(box_length, box_length) 
    yy, xx = grid[:,:,0], grid[:,:,1]
    peaks = 10 * torch.exp(-((xx - 0.34) ** 2 + (yy + 1.64) ** 2) / (2 * 0.4 ** 2))
    data = peaks + noise
    
    peak_detections = peak_local_max_2d(data, min_distance=1, threshold_abs=5)
    assert peak_detections.shape[0] > 1
    peak_detections = peak_local_max_2d(data, min_distance=5, threshold_abs=5)
    assert peak_detections.shape[0] == 1
    assert torch.max(peak_detections[0] - torch.tensor([34, 54])) < 2

