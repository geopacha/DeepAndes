from .moco_loader import load_moco_backbone
from .satmae_loader import backbone_builder, SatMAEForClassification

mae_backbone_builder = backbone_builder
mae_classifier = SatMAEForClassification

__all__ = ["load_moco_backbone", "mae_backbone_builder", "mae_classifier"]