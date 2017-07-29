"""Utility to derive calibration matrix from training examples."""

import numpy
import scipy.optimize


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

            pose_matrix = numpy.array([
                [1.0, 0.0, 0.0, x.x_offset_right],
                [0.0, 1.0, 0.0, 0.0],
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

            total_error += numpy.sum(error_matrix)

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
        method='Powell',
    )
    print(result)
