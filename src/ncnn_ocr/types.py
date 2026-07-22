"""Public types for ncnn-ocr."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class LineResult:
    """One detected and recognized text line."""

    text: str
    confidence: float
    box: tuple[Point, Point, Point, Point]
    box_score: float = 0.0
    rotated: bool = False
    angle_score: float = 0.0

    @classmethod
    def from_native(cls, item: dict) -> LineResult:
        raw_box = item.get("box") or ()
        points: list[Point] = []
        for entry in raw_box:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                points.append(Point(int(entry[0]), int(entry[1])))
        while len(points) < 4:
            points.append(Point(0, 0))
        return cls(
            text=str(item.get("text") or ""),
            confidence=float(item.get("confidence") or 0.0),
            box=(points[0], points[1], points[2], points[3]),
            box_score=float(item.get("box_score") or 0.0),
            rotated=bool(item.get("rotated")),
            angle_score=float(item.get("angle_score") or 0.0),
        )
