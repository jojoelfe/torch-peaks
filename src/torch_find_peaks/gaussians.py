import torch
import torch.nn as nn

class Gaussian2D(nn.Module):
    def __init__(self,
                 amplitude=1.,
                 x0=0.,
                 y0=0.,
                 sigma_x=1.,
                 sigma_y=1.):
        super(Gaussian2D, self).__init__()
        self.amplitude = nn.Parameter(torch.tensor(amplitude))
        self.x0 = nn.Parameter(torch.tensor(x0))
        self.y0 = nn.Parameter(torch.tensor(y0))
        self.sigma_x = nn.Parameter(torch.tensor(sigma_x))
        self.sigma_y = nn.Parameter(torch.tensor(sigma_y))
    
    def forward(self, x, y):
        return self.amplitude * torch.exp(
            -((x - self.x0) ** 2 / (2 * self.sigma_x ** 2) +
              (y - self.y0) ** 2 / (2 * self.sigma_y ** 2))
        )

class WarpedGaussian2D(nn.Module):
    def __init__(self,
                 amplitude=1.,
                 x0=0.,
                 y0=0.,
                 sigma_x=1.,
                 sigma_y=1.,
                 warp=1.,
                 warp_angle=0.):
        super(WarpedGaussian2D, self).__init__()
        self.amplitude = nn.Parameter(torch.tensor(amplitude))
        self.x0 = nn.Parameter(torch.tensor(x0))
        self.y0 = nn.Parameter(torch.tensor(y0))
        self.sigma_x = nn.Parameter(torch.tensor(sigma_x))
        self.sigma_y = nn.Parameter(torch.tensor(sigma_y))
        self.warp = nn.Parameter(torch.tensor(warp))
        self.warp_angle = nn.Parameter(torch.tensor(warp_angle))

    def forward(self, x, y):
        u = (x - self.x0) * torch.cos(self.warp_angle) - (y - self.y0) * torch.sin(self.warp_angle)
        v = (x - self.x0) * torch.sin(self.warp_angle) + (y - self.y0) * torch.cos(self.warp_angle)
        return self.amplitude * torch.exp(
            -((u - self.warp * v **2) ** 2 / (2 * self.sigma_x ** 2) +
               v  ** 2 / (2 * self.sigma_y ** 2))
        )

class Gaussian2DList(nn.Module):
    def __init__(self, num_gaussians, amplitude=1., x0=0., y0=0., sigma_x=1., sigma_y=1.):
        super(Gaussian2DList, self).__init__()
        self.amplitude = nn.Parameter(torch.ones(num_gaussians) * amplitude)
        self.x0 = nn.Parameter(torch.zeros(num_gaussians) + x0)
        self.y0 = nn.Parameter(torch.zeros(num_gaussians) + y0)
        self.sigma_x = nn.Parameter(torch.ones(num_gaussians) * sigma_x)
        self.sigma_y = nn.Parameter(torch.ones(num_gaussians) * sigma_y)

    def forward(self, x, y):
        """
        Forward pass for 2D Gaussian list.
        
        Args:
            x: x-coordinates
            y: y-coordinates
            
        Returns:
            Tensor of Gaussian values
        """
        x = x.unsqueeze(0)  # Shape: (num_points, num_gaussians)
        y = y.unsqueeze(0)
        
        return self.amplitude.unsqueeze(1) * torch.exp(
            -((x - self.x0.unsqueeze(1)) ** 2 / (2 * self.sigma_x.unsqueeze(1) ** 2) +
              (y - self.y0.unsqueeze(1)) ** 2 / (2 * self.sigma_y.unsqueeze(1) ** 2))
        )

class Gaussian3DList(nn.Module):
    def __init__(self, num_gaussians,sign=1):
        super(Gaussian3DList, self).__init__()
        self.amplitude = nn.Parameter(torch.ones(num_gaussians)*sign)
        self.x0 = nn.Parameter(torch.zeros(num_gaussians))
        self.y0 = nn.Parameter(torch.zeros(num_gaussians))
        self.z0 = nn.Parameter(torch.zeros(num_gaussians))
        self.sigma_x = nn.Parameter(torch.ones(num_gaussians))
        self.sigma_y = nn.Parameter(torch.ones(num_gaussians))
        self.sigma_z = nn.Parameter(torch.ones(num_gaussians))

    def forward(self, x, y, z):
        
        x = x.unsqueeze(0)  # Shape: (num_points, num_gaussians)
        y = y.unsqueeze(0)
        z = z.unsqueeze(0)
        return self.amplitude.unsqueeze(1) * torch.exp(
            -((x - self.x0.unsqueeze(1)) ** 2 / (2 * self.sigma_x.unsqueeze(1) ** 2) +
              (y - self.y0.unsqueeze(1)) ** 2 / (2 * self.sigma_y.unsqueeze(1) ** 2) +
              (z - self.z0.unsqueeze(1)) ** 2 / (2 * self.sigma_z.unsqueeze(1) ** 2))
        )  