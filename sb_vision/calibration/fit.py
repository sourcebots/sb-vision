"""Utility to derive calibration matrix from training examples."""

import collections

import numpy
import numpy.linalg
import sklearn.base
import sklearn.metrics
import sklearn.pipeline
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.model_selection

from ..tokens import homography_matrix_to_distance_model_input_vector

Calibration = collections.namedtuple('Calibration', (
    'resolution',
    'z_model',
    'x_model',
))


def _find_coefficients(X, y, name):
    pipeline = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.RobustScaler(),
        sklearn.linear_model.LinearRegression(),
    )

    pipeline.fit(X, y)

    print(
        "MAE({}): ".format(name),
        sklearn.metrics.mean_absolute_error(
            y,
            pipeline.predict(X),
        ),
    )

    scaler = pipeline.steps[0][1]
    regressor = pipeline.steps[1][1]

    intercept = regressor.intercept_.ravel()[0]
    coefs = regressor.coef_.ravel() / scaler.scale_
    biases = -scaler.center_.ravel()

    return {
        'model': 'qr',
        'intercept': intercept,
        'coefs': coefs,
        'biases': biases,
    }


def fit(training_examples):
    """Fit calibration matrix to given iterable of training examples."""
    training_examples = list(training_examples)

    X = numpy.array([
        homography_matrix_to_distance_model_input_vector(
            x.homography_matrix,
        )
        for x in training_examples
    ])

    y_z = numpy.array([x.z_distance for x in training_examples])
    y_x = numpy.array([x.x_offset_right for x in training_examples])

    model_x = _find_coefficients(X, y_x, 'x')
    model_z = _find_coefficients(X, y_z, 'z')

    return Calibration(
        resolution=training_examples[0].size,
        z_model=model_z,
        x_model=model_x,
    )
