"""Networks for torch."""

from .randlanet import RandLANet
from .unet import UNet
from .siamesenet_acf import SiameseNetAcf

__all__ = ['RandLANet', 'UNet', 'SiameseNetAcf']

try:
    from .openvino_model import OpenVINOModel
    __all__.append("OpenVINOModel")
except Exception:
    pass
