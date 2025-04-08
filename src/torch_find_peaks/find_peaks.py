import einops
import torch
import torch.nn.functional as F

def find_peaks_2d(
        image: torch.Tensor,
        min_distance: int = 1,
        threshold_abs: float = 0.0,
        exclude_border: int = 0,
):
    image = einops.rearrange(image, "h w -> 1 1 h w")
    mask = F.max_pool2d(
        image,
        kernel_size=(min_distance * 2 + 1, min_distance * 2 + 1),
        stride=1,
        padding=min_distance,
    )
    mask = einops.rearrange(mask, "1 1 h w -> h w")
    image = einops.rearrange(image, "1 1 h w -> h w")
    mask = (image == mask) & (image > threshold_abs)
    if exclude_border > 0:
        mask[:exclude_border, :] = False
        mask[-exclude_border:, :] = False
        mask[:, :exclude_border] = False
        mask[:, -exclude_border:] = False
    
    coords = torch.nonzero(mask, as_tuple=False)
    return coords

def find_peaks_3d(
        volume: torch.Tensor,
        min_distance: int = 1,
        threshold_abs: float = 0.0,
        exclude_border: int = 0,
):
    volume = einops.rearrange(volume, "d h w -> 1 1 d h w")
    mask = F.max_pool3d(
        volume,
        kernel_size=(min_distance * 2 + 1, min_distance * 2 + 1, min_distance * 2 + 1),
        stride=1,
        padding=min_distance,
    )
    mask = einops.rearrange(mask, "1 1 d h w -> d h w")
    volume = einops.rearrange(volume, "1 1 d h w -> d h w")
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
