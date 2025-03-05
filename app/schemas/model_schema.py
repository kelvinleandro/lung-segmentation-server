from typing import List

from pydantic import BaseModel


class Point(BaseModel):
    x: float
    y: float


class SegmentedPointsResponse(BaseModel):
    points: List[Point]
