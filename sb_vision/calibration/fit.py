"""Utility to derive calibration matrix from training examples."""

import numpy
import numpy.linalg
import scipy.linalg
import scipy.optimize
import sklearn.base
import sklearn.metrics
import sklearn.pipeline
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.model_selection
import collections

Calibration = collections.namedtuple('Calibration', (
    'resolution',
    'z_model',
    'x_model',
))


def fit(training_examples):
    """Fit calibration matrix to given iterable of training examples."""

    training_examples = list(training_examples)

    pipeline = sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.PolynomialFeatures(
            degree=2,
            interaction_only=False,
            include_bias=False,
        ),
        sklearn.preprocessing.RobustScaler(),
        sklearn.linear_model.LinearRegression(),
    )

    X = numpy.array([
        x.homography_matrix.ravel()
        for x in training_examples
    ])

    y_z = numpy.array([x.z_distance for x in training_examples])
    y_x = numpy.array([x.x_offset_right for x in training_examples])

    pipeline_z = sklearn.base.clone(pipeline)
    pipeline_z.fit(X, y_z)
    print("MAE(z): ", sklearn.metrics.mean_absolute_error(
        y_z,
        pipeline_z.predict(X),
    ))

    pipeline_x = sklearn.base.clone(pipeline)
    pipeline_x.fit(X, y_x)
    print("MAE(x): ", sklearn.metrics.mean_absolute_error(
        y_x,
        pipeline_x.predict(X),
    ))

    return Calibration(
        resolution=training_examples[0].size,
        z_model=pipeline_z,
        x_model=pipeline_x,
    )
