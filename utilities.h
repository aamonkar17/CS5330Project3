#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

int getEmbedding(cv::Mat& src, cv::Mat& embedding, cv::dnn::Net& net, int debug);

void prepEmbeddingImage(cv::Mat& frame, cv::Mat& embimage,
    int cx, int cy, float theta,
    float minE1, float maxE1,
    float minE2, float maxE2,
    int debug);
