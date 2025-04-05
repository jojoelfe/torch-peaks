import torch
import torch.nn.functional as F

def peak_local_max_2d(
        image: torch.Tensor,
        min_distance: int = 1,
        threshold_abs: float = 0.0,
        exclude_border: bool = True,
):
    return F.max_pool2d(
        image.unsqueeze(0).unsqueeze(0),
        kernel_size=(min_distance * 2 + 1, min_distance * 2 + 1),
        stride=1,
        padding=min_distance,
    ).squeeze(0).squeeze(0) 

if __name__ == "__main__":
    # Example usage
    image = torch.randn( 5, 5)
    print(image)
    peaks = peak_local_max_2d(image, min_distance=2, threshold_abs=0.5)
    print(peaks)