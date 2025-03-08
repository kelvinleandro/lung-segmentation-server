from pydantic import BaseModel
from typing import List

class Point(BaseModel):
    x: float
    y: float

class SegmentedPointsResponse(BaseModel):
    points: List[Point]