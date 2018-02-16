#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <string>

extern "C" {
    void* cvopen(const int device_id);
    void cvclose(void* context);
    int cvcapture(void* context, void* buffer, size_t width, size_t height);

    void cvthreshold(
        const void* src_buffer,
        void* dst_buffer,
        size_t width,
        size_t height
    );
}

#include "opencv2/opencv.hpp"

void skipframes(cv::VideoCapture* context, int frames_to_skip) {
    for (int i=0; i < frames_to_skip; i++) {
        context->grab();
    }
}

void warmup(cv::VideoCapture* context) {
    // The TeckNet camera needs about 11 frames to warm up either when first
    // opened or when changing resolution. Without this the images returned are
    // too dark to be useful.
    skipframes(context, 11);
}

void* cvopen(const int device_id) {
    cv::VideoCapture* context = new cv::VideoCapture(device_id);
    if (!context->isOpened()) {
        delete context;
        return NULL;
    }
    warmup(context);
    return reinterpret_cast<void*>(context);
}

void cvclose(void* context) {
    cv::VideoCapture* cap = reinterpret_cast<cv::VideoCapture*>(context);
    delete cap;
}

int cvcapture(void* context, void* buffer, size_t width, size_t height) {
    cv::VideoCapture* cap = reinterpret_cast<cv::VideoCapture*>(context);

    double current_width = cap->get(CV_CAP_PROP_FRAME_WIDTH);
    double current_height = cap->get(CV_CAP_PROP_FRAME_HEIGHT);

    if (current_width != (double)width || current_height != (double)height) {
        fprintf( stderr, "Changing resolution from %dx%d to %dx%d\n", current_width, current_height, width, height);
        cap->set(CV_CAP_PROP_FRAME_WIDTH, width);
        if (cap->get(CV_CAP_PROP_FRAME_HEIGHT) != (double)height) {
            cap->set(CV_CAP_PROP_FRAME_HEIGHT, height);
        }

        // Get the camera warmed up for the new resolution
        warmup(cap);
    }
    else {
        // To be sure that we get an image which accurately describes what is in
        // front of the camera _right now_ (rather than whenever the last frames
        // were grabbed) we ditch the last few frames.
        // This is needed with the TeckNet cameras we're using, though may not
        // be needed for others. Note: manual testing suggests that skipping 4
        // frames works for this purpose, though we deliberatly skip one more
        // than that to reduce the chances that we'll get a bad frame (in case
        // this is timing related).
        skipframes(cap, 5);
    }

    if (cap->get(CV_CAP_PROP_FRAME_WIDTH) != (double)width) {
        fprintf(stderr, "Incorrect width set on cap: %f\n", cap->get(CV_CAP_PROP_FRAME_WIDTH));
        return 0;
    }

    if (cap->get(CV_CAP_PROP_FRAME_HEIGHT) != (double)height) {
        fprintf(stderr, "Incorrect height set on cap: %f\n", cap->get(CV_CAP_PROP_FRAME_HEIGHT));
        return 0;
    }

    cv::Mat colour_image, greyscale_image;

    (*cap) >> colour_image;
    if (colour_image.empty()) {
        fprintf(stderr, "Failed to capture image (result was empty)\n");
        return 0;
    }

    cv::cvtColor(colour_image, greyscale_image, cv::COLOR_BGR2GRAY);

    if (!greyscale_image.isContinuous()) {
        return 0;
    }
    int died_horribly = 0;
    if (greyscale_image.size().width != width) {
        fprintf(
            stderr,
            "Width mismatch: %d expected, %d actual\n",
            width,
            greyscale_image.size().width
        );
        died_horribly = 1;
    }
    if (greyscale_image.size().height != height) {
        fprintf(
            stderr,
            "Height mismatch: %d expected, %d actual\n",
            height,
            greyscale_image.size().height
        );
        died_horribly = 1;
    }
    if (died_horribly) {
        return 0;
    }

    memcpy(
        buffer,
        greyscale_image.ptr(),
        width * height
    );
    return 1;
}

void cvthreshold(
    const void* src_buffer,
    void* dst_buffer,
    size_t width,
    size_t height
) {
    cv::Mat src(
        static_cast<int>(height),
        static_cast<int>(width),
        CV_8UC1
    );
    cv::Mat denoised;
    cv::Mat thresholded;

    memcpy(
        src.ptr(),
        src_buffer,
        width * height
    );

    cv::medianBlur(src, denoised, 3);

    size_t kernel_size = (width < height) ? width : height;
    kernel_size /= 2;
    kernel_size |= 1;

    cv::adaptiveThreshold(
        denoised,
        thresholded,
        255.0,
        CV_ADAPTIVE_THRESH_MEAN_C,
        CV_THRESH_BINARY,
        kernel_size,
        0.0
    );

    memcpy(
        dst_buffer,
        thresholded.ptr(),
        width * height
    );
}
