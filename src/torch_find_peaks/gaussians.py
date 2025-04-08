import einops
import torch
import torch.nn as nn

#class WarpedGaussian2D(nn.Module):
#    def __init__(self,
#                 amplitude=1.,
#                 x0=0.,
#                 y0=0.,
#                 sigma_x=1.,
#                 sigma_y=1.,
#                 warp=1.,
#                 warp_angle=0.):
#        super(WarpedGaussian2D, self).__init__()
#        self.amplitude = nn.Parameter(torch.tensor(amplitude))
#        self.x0 = nn.Parameter(torch.tensor(x0))
#        self.y0 = nn.Parameter(torch.tensor(y0))
#        self.sigma_x = nn.Parameter(torch.tensor(sigma_x))
#        self.sigma_y = nn.Parameter(torch.tensor(sigma_y))
#        self.warp = nn.Parameter(torch.tensor(warp))
#        self.warp_angle = nn.Parameter(torch.tensor(warp_angle))
#
#    def forward(self, x, y):
#        u = (x - self.x0) * torch.cos(self.warp_angle) - (y - self.y0) * torch.sin(self.warp_angle)
#        v = (x - self.x0) * torch.sin(self.warp_angle) + (y - self.y0) * torch.cos(self.warp_angle)
#        return self.amplitude * torch.exp(
#            -((u - self.warp * v **2) ** 2 / (2 * self.sigma_x ** 2) +
#               v  ** 2 / (2 * self.sigma_y ** 2))
#        )

class Gaussian2D(nn.Module):
    def __init__(self, 
                 amplitude: torch.tensor = torch.tensor([1.0]), 
                 center_x: torch.tensor = torch.tensor([0.0]),
                 center_y: torch.tensor = torch.tensor([0.0]),
                 sigma_x: torch.tensor = torch.tensor([1.0]),
                 sigma_y: torch.tensor = torch.tensor([1.0])
    ):
        super(Gaussian2D, self).__init__()
        # Ensure that the parameters are tensors of equal dimensions
        assert amplitude.shape == center_x.shape == center_y.shape == sigma_x.shape == sigma_y.shape, \
            "All parameters must have the same shape."

        self.amplitude = nn.Parameter(amplitude)
        self.center_x = nn.Parameter(center_x)
        self.center_y = nn.Parameter(center_y)
        self.sigma_x = nn.Parameter(sigma_x)
        self.sigma_y = nn.Parameter(sigma_y)

    def forward(self, grid):
        """
        Forward pass for 2D Gaussian list.
        
        Args:
            grid: Tensor of shape (h,w, 2) containing 2D coordinates.
            
        Returns:
            Tensor of Gaussian values
        """
        amplitude = einops.rearrange(self.amplitude, '... -> 1 1 ...')
        center_x = einops.rearrange(self.center_x, '... -> 1 1 ...')
        center_y = einops.rearrange(self.center_y, '... -> 1 1 ...')
        sigma_x = einops.rearrange(self.sigma_x, '... -> 1 1 ...')
        sigma_y = einops.rearrange(self.sigma_y, '... -> 1 1 ...')

        grid_x = einops.rearrange(grid[..., 1], 'h w -> h w 1')
        grid_y = einops.rearrange(grid[..., 0], 'h w -> h w 1')

        gaussian = amplitude * torch.exp(
            -((grid_x - center_x) ** 2 / (2 * sigma_x ** 2) +
              (grid_y - center_y) ** 2 / (2 * sigma_y ** 2))
        )

        return einops.rearrange(gaussian, 'h w ... -> ... h w')
