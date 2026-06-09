"""Feature extractors. Each writes `<name>_t<track>.csv` to the cache, keyed by participant,video.

    from errhri_features.extractors import ActionUnitExtractor
    ActionUnitExtractor(s=10, workers=8).run(track=1)
"""
from .base import BaseExtractor
from .action_units import ActionUnitExtractor
from .audio import AudioExtractor
from .embedding import EmbeddingExtractor
from .blendshape import BlendshapeExtractor
from .trajectory import TrajectoryExtractor

REGISTRY = {
    "au": ActionUnitExtractor,
    "audio": AudioExtractor,
    "embed": EmbeddingExtractor,
    "blend": BlendshapeExtractor,
    "traj": TrajectoryExtractor,
}

__all__ = ["BaseExtractor", "ActionUnitExtractor", "AudioExtractor", "EmbeddingExtractor",
           "BlendshapeExtractor", "TrajectoryExtractor", "REGISTRY"]
