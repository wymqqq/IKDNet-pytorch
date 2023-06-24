"""3D ML pipelines for torch."""

from .semantic_segmentation import SemanticSegmentation
from .semantic_segmentation_img import SemanticSegmentationImg
from .semantic_segmentation_dual import SemanticSegmentationDual
from .object_detection import ObjectDetection

__all__ = ['SemanticSegmentation', 'SemanticSegmentationImg', 'SemanticSegmentationDual', 'ObjectDetection']
