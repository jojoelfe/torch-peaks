import torch
import torch.nn.functional as F

def peak_local_max_2d(
        image: torch.Tensor,
        min_distance: int = 1,
        threshold_abs: float = 0.0,
        exclude_border: int = 0,
):
    mask = F.max_pool2d(
        image.unsqueeze(0).unsqueeze(0),
        kernel_size=(min_distance * 2 + 1, min_distance * 2 + 1),
        stride=1,
        padding=min_distance,
    ).squeeze(0).squeeze(0) 

    mask = (image == mask) & (image > threshold_abs)

    if exclude_border > 0:
        mask[:exclude_border, :] = False
        mask[-exclude_border:, :] = False
        mask[:, :exclude_border] = False
        mask[:, -exclude_border:] = False
    
    coords = torch.nonzero(mask, as_tuple=False)
    return coords

def peak_local_max_3d(
        volume: torch.Tensor,
        min_distance: int = 1,
        threshold_abs: float = 0.0,
        exclude_border: int = 0,
):

    mask = F.max_pool3d(
        volume.unsqueeze(0).unsqueeze(0),
        kernel_size=(min_distance * 2 + 1, min_distance * 2 + 1, min_distance * 2 + 1),
        stride=1,
        padding=min_distance,
    ).squeeze(0).squeeze(0)
    mask = (volume == mask) & (volume > threshold_abs)
    if exclude_border > 0:
        mask[:exclude_border, :, :] = False
        mask[-exclude_border:, :, :] = False
        mask[:, :exclude_border, :] = False
        mask[:, -exclude_border:, :] = False
        mask[:, :, :exclude_border] = False
        mask[:, :, -exclude_border:] = False

    coords = torch.nonzero(mask, as_tuple=False)
    return coords
