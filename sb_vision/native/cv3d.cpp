#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <string>
#include <vector>

extern "C" {
    int solve_pnp(
        const float object_points[],
        const float image_points[],
        const float camera_matrix[],
        const float dist_coeffs[],
        float rvec[],
        float tvec[]
    );
}

#include "opencv2/opencv.hpp"

#define COPY_INTO(mat, data) \
    memcpy(mat.ptr(), data, mat.rows * mat.cols * sizeof(float));


int solve_pnp(
    const float object_points[],
    const float image_points[],
    const float camera_matrix[],
    const float dist_coeffs[],
    float rvec[],
    float tvec[]
) {
    cv::Mat object_points_mat(4, 3, CV_32FC1);
    COPY_INTO(object_points_mat, object_points);

    cv::Mat image_points_mat(4, 2, CV_32FC1);
    COPY_INTO(image_points_mat, image_points);

    cv::Mat camera_matrix_mat(3, 3, CV_32FC1);
    COPY_INTO(camera_matrix_mat, camera_matrix);

    cv::Mat dist_coeffs_mat(5, 1, CV_32FC1);
    COPY_INTO(dist_coeffs_mat, dist_coeffs);

    cv::Mat rvec_mat(3, 1, CV_32FC1);
    cv::Mat tvec_mat(3, 1, CV_32FC1);

    bool ret = cv::solvePnP(
        object_points_mat,
        image_points_mat,
        camera_matrix_mat,
        dist_coeffs_mat,
        rvec_mat,
        tvec_mat
    );

    memcpy(rvec, rvec_mat.ptr(), 3 * sizeof(float));
    memcpy(tvec, tvec_mat.ptr(), 3 * sizeof(float));

    // OpenCV returns the 'y' coordinate positive downwards, yet we want
    // positive meaning upwards
    tvec[1] = -tvec[1];

    return (int) ret;
}
