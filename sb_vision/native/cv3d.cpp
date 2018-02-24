#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <string>

extern "C" {
    int solve_pnp(
        const void* object_points,
        const void* image_points,
        const void* camera_matrix,
        const void* dist_coeffs,
        void* rvec,
        void* tvec
    );
}

#include "opencv2/opencv.hpp"

int solve_pnp(
    const void* object_points,
    const void* image_points,
    const void* camera_matrix,
    const void* dist_coeffs,
    void* rvec,
    void* tvec
) {
    return 1;
}
