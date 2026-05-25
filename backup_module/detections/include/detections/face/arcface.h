#pragma once
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

#include <cmath>
#include <cstring>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class ArcFace {
public:
    ArcFace();

    std::string const BASE_PATH = "model/face/";
    std::string const MODEL_FILE = "arcface.tflite";

    std::vector<float> calc_embedding(const cv::Mat& img);
    float get_distance_embeddings(const std::vector<float>& emb1, const std::vector<float>& emb2);

private:
    void l2_norm(std::vector<float>& vec);
    std::unique_ptr<tflite::Interpreter> interpreter;
    std::unique_ptr<tflite::FlatBufferModel> model;
};

float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);