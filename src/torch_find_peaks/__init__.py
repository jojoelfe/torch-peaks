"""Peak finding and fitting using torch"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-find-peaks")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Johannes Elferich"
__email__ = "jojotux123@hotmail.com"

from .find_peaks import find_peaks_2d, find_peaks_3d
from .gaussians import Gaussian2D, WarpedGaussian2D, Gaussian3DList, Gaussian2DList
from .fit_gaussians import fit_gaussians_2d
__all__ = [
    
]
