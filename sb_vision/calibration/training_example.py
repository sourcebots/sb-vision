"""Definitions for training examples."""

import collections


TrainingExample = collections.namedtuple('TrainingExample', (
    'image_file',
    'z_distance',
    'x_offset_right',
))
