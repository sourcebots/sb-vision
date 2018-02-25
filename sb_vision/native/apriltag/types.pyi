from typing import List, NamedTuple

class MatrixData:
    ncols: int
    nrows: int
    data: List[float]


class ApriltagDetection:
    id: int
    hamming: int
    goodness: float
    decision_margin: float
    H: MatrixData
    p: List[List[float]]
