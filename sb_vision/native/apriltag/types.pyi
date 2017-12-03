from typing import List, NamedTuple

MatrixData = NamedTuple('MatrixData', [
    ('ncols', int),
    ('nrows', int),
    ('data', List[float]),
])


ApriltagDetection = NamedTuple('ApriltagDetection', [
    ('id', int),
    ('hamming', int),
    ('goodness', float),
    ('decision_margin', float),
    ('H', MatrixData),
    # Other fields omitted
])
