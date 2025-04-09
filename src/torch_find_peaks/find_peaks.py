import einops
import torch
import torch.nn.functional as F


def find_peaks_2d(
        image: torch.Tensor,
        min_distance: int = 1,
        threshold_abs: float = 0.0,
        exclude_border: int = 0,
) -> torch.Tensor:
    """
    Find local peaks in a 2D image.

    Parameters
    ----------
    image : torch.Tensor
        A 2D tensor representing the input image.
    min_distance : int, optional
        Minimum distance between peaks. Default is 1.
    threshold_abs : float, optional
        Minimum intensity value for a peak to be considered. Default is 0.0.
    exclude_border : int, optional
        Width of the border to exclude from peak detection. Default is 0.

    Returns
    -------
    torch.Tensor
        A tensor of shape (N, 2), where N is the number of peaks, and each row
        contains the (Y, X) coordinates of a peak.
    """
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
) -> torch.Tensor:
    """
    Find local peaks in a 3D volume.

    Parameters
    ----------
    volume : torch.Tensor
        A 3D tensor representing the input volume.
    min_distance : int, optional
        Minimum distance between peaks. Default is 1.
    threshold_abs : float, optional
        Minimum intensity value for a peak to be considered. Default is 0.0.
    exclude_border : int, optional
        Width of the border to exclude from peak detection. Default is 0.

    Returns
    -------
    torch.Tensor
        A tensor of shape (N, 3), where N is the number of peaks, and each row
        contains the (Z, Y, X) coordinates of a peak.
    """
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
