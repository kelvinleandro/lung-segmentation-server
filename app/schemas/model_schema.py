from typing import List, Dict, Any

from pydantic import BaseModel


class Point(BaseModel):
    x: float
    y: float


class SegmentedPointsResponse(BaseModel):
    points: List[Point]

class SegmentationRequest(BaseModel):
    preprocessing_params: Dict[str, Any]
    segmentation_params: Dict[str, Any]
    postprocessing_params: Dict[str, Any]