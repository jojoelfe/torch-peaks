"""Peak finding and fitting using torch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-peaks")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Johannes Elferich"
__email__ = "jojotux123@hotmail.com"

from .find_peaks import peak_local_max_2d, peak_local_max_3d
from .gaussians import Gaussian2D, WarpedGaussian2D, Gaussian3DList, Gaussian2DList
from .fit_gaussians import fit_gaussians_2d, fit_gaussians_3d

__all__ = [
    'peak_local_max_2d',
    'peak_local_max_3d',
    'Gaussian2D',
    'WarpedGaussian2D',
    'Gaussian3DList',
    'Gaussian2DList',
    'fit_gaussians_2d',
    'fit_gaussians_3d',
]
