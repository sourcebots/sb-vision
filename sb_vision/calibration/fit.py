"""Utility to derive calibration matrix from training examples."""

import numpy
import scipy.optimize
import collections

Calibration = collections.namedtuple('Calibration', (
    'focal_length',
    'principal_x',
    'principal_y',
    'skew',
))


def fit(training_examples):
    """Fit calibration matrix to given iterable of training examples."""

    training_examples = list(training_examples)

    def reconstruction_error(x):
        [focal_length, principal_x, principal_y, skew] = x

        total_error = 0.0

        for x in training_examples:
            calibration_matrix = numpy.array([
                [x.size[0] * focal_length, skew, principal_x * x.size[0]],
                [0.0, x.size[1] * focal_length, principal_y * x.size[1]],
                [0.0, 0.0, 1.0],
            ])

            error_levels = []

            for rotation in (
                (1, 0, 0, 1),
                (0, -1, 1, 0),
                (-1, 0, 0, -1),
                (0, 1, -1, 0),
            ):
                a, b, c, d = rotation

                pose_matrix = numpy.array([
                    [a, b, 0.0, x.x_offset_right],
                    [c, d, 0.0, 0.0],
                    [0.0, 0.0, 1.0, x.z_distance],
                ])

                homography_matrix_with_extra_col = numpy.array([
                    x.homography_matrix[:, 0],
                    x.homography_matrix[:, 1],
                    numpy.cross(
                        x.homography_matrix[:, 0],
                        x.homography_matrix[:, 1],
                    ),
                    x.homography_matrix[:, 2],
                ]).T

                reconstructed_homography_matrix = calibration_matrix.dot(
                    pose_matrix,
                )

                print(homography_matrix_with_extra_col)
                print(reconstructed_homography_matrix)

                error_matrix = abs(
                    reconstructed_homography_matrix -
                    homography_matrix_with_extra_col
                )

                error_levels.append(numpy.sum(error_matrix[:, 3]))

            total_error += max(error_levels)

        return total_error

    initial_focal_length = 1.0  # 1m
    initial_skew = 0
    initial_principal_x = 0.5
    initial_principal_y = 0.5

    result = scipy.optimize.minimize(
        reconstruction_error,
        x0=[
            initial_focal_length,
            initial_principal_x,
            initial_principal_y,
            initial_skew,
        ],
        method='Nelder-Mead',
    )
    print(result)

    final_focal_length, final_principal_x, final_principal_y, final_skew = \
        result.x

    return Calibration(
        focal_length=final_focal_length,
        skew=final_skew,
        principal_x=final_principal_x,
        principal_y=final_principal_y,
    )