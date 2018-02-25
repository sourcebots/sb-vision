#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <string>
#include <vector>

extern "C" {
    int solve_pnp(
        const double object_points[],
        const double image_points[],
        const double camera_matrix[],
        const double dist_coeffs[],
        double rvec[],
        double tvec[]
    );
}

#include "opencv2/opencv.hpp"

void print_array(const int rows, const int cols, const double values[]) {
    printf("[\n");
    for (int i=0; i<rows; i++) {
        printf("  [");
        for (int j=0; j<cols; j++) {
            int idx = (i * cols) + j;
            printf("%f", values[idx]);
            if (j+1 < cols) {
                printf(", ");
            }
        }
        printf("],\n");
    }
    printf("]\n");
}

void print_mat(cv::Mat& mat) {
    printf("(%d x %d = %d) ", mat.rows, mat.cols, mat.total());
    printf("[\n");
    for (int i=0; i<mat.rows; i++) {
        printf("  [");
        for (int j=0; j<mat.cols; j++) {
            double * val = (double *)mat.ptr(i, j);
            printf("%f", *val);
            if (j+1 < mat.cols) {
                printf(", ");
            }
        }
        printf("],\n");
    }
    printf("]\n");
}

#define COPY_INTO(mat, data) \
    memcpy(mat.ptr(), data, mat.rows * mat.cols * sizeof(double));


int solve_pnp(
    const double object_points[],
    const double image_points[],
    const double camera_matrix[],
    const double dist_coeffs[],
    double rvec[],
    double tvec[]
) {
    // Copy our input data into OpenCV's Mat wrapper
    cv::Mat object_points_mat(4, 3, CV_64FC1);
    COPY_INTO(object_points_mat, object_points);

    cv::Mat image_points_mat(4, 2, CV_64FC1);
    COPY_INTO(image_points_mat, image_points);

    cv::Mat camera_matrix_mat(3, 3, CV_64FC1);
    COPY_INTO(camera_matrix_mat, camera_matrix);

    cv::Mat dist_coeffs_mat(5, 1, CV_64FC1);
    COPY_INTO(dist_coeffs_mat, dist_coeffs);

    // These are our output data
    cv::Mat rvec_mat, tvec_mat;

    bool ret = cv::solvePnP(
        object_points_mat,
        image_points_mat,
        camera_matrix_mat,
        dist_coeffs_mat,
        rvec_mat,
        tvec_mat
    );

    // Copy the output data out of OpenCV's wrappers
    memcpy(rvec, rvec_mat.ptr(), 3 * sizeof(double));
    memcpy(tvec, tvec_mat.ptr(), 3 * sizeof(double));

    // OpenCV returns the 'y' coordinate positive downwards, yet we want
    // positive meaning upwards
    tvec[1] = -tvec[1];

    return (int) ret;
}
