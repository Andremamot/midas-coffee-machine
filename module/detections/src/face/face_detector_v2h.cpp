#ifdef V2H
#include <detections/face/face_detector_v2h.h>

using Clock = std::chrono::steady_clock;
static inline long ms_between(const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
}

FaceDetectorV2H::FaceDetectorV2H() {}

FaceDetectorV2H::~FaceDetectorV2H() {
    if (drpai_output_buf) {
        delete[] drpai_output_buf;
    }
}

double FaceDetectorV2H::sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

float FaceDetectorV2H::float16_to_float32(uint16_t a) {
    return __extendXfYf2__<uint16_t, uint16_t, 10, float, uint32_t, 23>(a);
}

int32_t FaceDetectorV2H::yolo_index(uint8_t n, int32_t offs, int32_t channel) {
    uint8_t num_grid = c::yolov3::num_grids[n];
    return offs + channel * num_grid * num_grid;
}

int32_t FaceDetectorV2H::yolo_offset(uint8_t n, int32_t b, int32_t y, int32_t x) {
    uint8_t num = c::yolov3::num_grids[n];
    uint32_t prev_layer_num = 0;
    int32_t i = 0;

    for (i = 0; i < n; i++) {
        prev_layer_num += c::yolov3::NUM_BB * (c::yolov3::NUM_CLASS + 5) * c::yolov3::num_grids[i] *
                          c::yolov3::num_grids[i];
    }
    return prev_layer_num + b * (c::yolov3::NUM_CLASS + 5) * num * num + y * num + x;
}

void FaceDetectorV2H::R_Post_Proc(float* floatarr) {
    /* Following variables are required for correct_region_boxes in Darknet
     * implementation*/
    /* Note: This implementation refers to the "darknet detector test" */

    float new_w, new_h;
    float correct_w = 1.;
    float correct_h = 1.;
    if ((float)(c::yolov3::MODEL_IN_W / correct_w) < (float)(c::yolov3::MODEL_IN_H / correct_h)) {
        new_w = (float)c::yolov3::MODEL_IN_W;
        new_h = correct_h * c::yolov3::MODEL_IN_W / correct_w;
    } else {
        new_w = correct_w * c::yolov3::MODEL_IN_H / correct_h;
        new_h = c::yolov3::MODEL_IN_H;
    }

    int32_t n = 0;
    int32_t b = 0;
    int32_t y = 0;
    int32_t x = 0;
    int32_t offs = 0;
    int32_t i = 0;
    float tx = 0;
    float ty = 0;
    float tw = 0;
    float th = 0;
    float tc = 0;
    float center_x = 0;
    float center_y = 0;
    float box_w = 0;
    float box_h = 0;
    float objectness = 0;
    uint8_t num_grid = 0;
    uint8_t anchor_offset = 0;
    float classes[c::yolov3::NUM_CLASS];
    float max_pred = 0;
    int32_t pred_class = -1;
    float probability = 0;
    Detection d;
    /* Clear the detected result list */
    det.clear();

    for (n = 0; n < c::yolov3::NUM_INF_OUT_LAYER; n++) {
        num_grid = c::yolov3::num_grids[n];
        anchor_offset = 2 * c::yolov3::NUM_BB * (c::yolov3::NUM_INF_OUT_LAYER - (n + 1));

        for (b = 0; b < c::yolov3::NUM_BB; b++) {
            for (y = 0; y < num_grid; y++) {
                for (x = 0; x < num_grid; x++) {
                    offs = yolo_offset(n, b, y, x);
                    tx = floatarr[offs];
                    ty = floatarr[yolo_index(n, offs, 1)];
                    tw = floatarr[yolo_index(n, offs, 2)];
                    th = floatarr[yolo_index(n, offs, 3)];
                    tc = floatarr[yolo_index(n, offs, 4)];

                    /* Compute the bounding box */
                    /*get_region_box*/
                    center_x = ((float)x + sigmoid(tx)) / (float)num_grid;
                    center_y = ((float)y + sigmoid(ty)) / (float)num_grid;

                    box_w = (float)exp(tw) * c::yolov3::anchors[anchor_offset + 2 * b + 0] /
                            (float)c::yolov3::MODEL_IN_W;
                    box_h = (float)exp(th) * c::yolov3::anchors[anchor_offset + 2 * b + 1] /
                            (float)c::yolov3::MODEL_IN_W;
                    /* Adjustment for VGA size */
                    /* correct_region_boxes */
                    center_x =
                        (center_x - (c::yolov3::MODEL_IN_W - new_w) / 2. / c::yolov3::MODEL_IN_W) /
                        ((float)new_w / c::yolov3::MODEL_IN_W);
                    center_y =
                        (center_y - (c::yolov3::MODEL_IN_H - new_h) / 2. / c::yolov3::MODEL_IN_H) /
                        ((float)new_h / c::yolov3::MODEL_IN_H);
                    box_w *= (float)(c::yolov3::MODEL_IN_W / new_w);
                    box_h *= (float)(c::yolov3::MODEL_IN_H / new_h);

                    center_x = round(center_x * DRPAI_IN_WIDTH);
                    center_y = round(center_y * DRPAI_IN_HEIGHT);
                    box_w = round(box_w * DRPAI_IN_WIDTH);
                    box_h = round(box_h * DRPAI_IN_HEIGHT);

                    objectness = sigmoid(tc);

                    Box bb = {center_x, center_y, box_w, box_h};

                    /* Get the class prediction associated with each BB  [5: ]
                     */
                    for (i = 0; i < c::yolov3::NUM_CLASS; i++) {
                        classes[i] = sigmoid(floatarr[yolo_index(n, offs, 5 + i)]);
                    }

                    max_pred = 0;
                    pred_class = -1;
                    /*Get the predicted class */
                    for (i = 0; i < c::yolov3::NUM_CLASS; i++) {
                        if (classes[i] > max_pred) {
                            pred_class = i;
                            max_pred = classes[i];
                        }
                    }

                    /* Store the result into the list if the probability is more
                     * than the threshold */
                    probability = max_pred * objectness;
                    if (probability > c::yolov3::TH_PROB) {
                        d.box = bb;
                        d.class_id = pred_class;
                        d.score = probability;
                        det.push_back(d);
                    }
                }
            }
        }
    }

    /* Non-Maximum Suppression filter */
    filter_boxes_nms(det, det.size(), c::yolov3::TH_NMS);

    std::vector<Detection> final_det;
    for (const auto& d : det)
        if (d.score > 0.0f) final_det.push_back(d);

    det = final_det;

    return;
}

std::tuple<std::vector<Detection>> FaceDetectorV2H::detect(cv::Mat& frame) {
    det.clear();
    cv::Size size(c::yolov3::MODEL_IN_H, c::yolov3::MODEL_IN_W);

    cv::Mat resized;

    // preprocess
    cv::resize(frame, resized, size);

    cv::Mat img_rgb;
    cv::cvtColor(resized, img_rgb, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> rgb_images;
    split(img_rgb, rgb_images);

    cv::Mat m_flat_r = rgb_images[0].reshape(1, 1);
    cv::Mat m_flat_g = rgb_images[1].reshape(1, 1);
    cv::Mat m_flat_b = rgb_images[2].reshape(1, 1);

    cv::Mat matArray[] = {m_flat_r, m_flat_g, m_flat_b};
    cv::Mat frameCHW;
    int ret = 0;

    hconcat(matArray, 3, frameCHW);

    /*convert to FP32*/
    frameCHW.convertTo(frameCHW, CV_32FC3);

    /* normailising  pixels */
    divide(frameCHW, 255.0, frameCHW);

    /* DRP AI input image should be continuous buffer */
    if (!frameCHW.isContinuous()) frameCHW = frameCHW.clone();

    cv::Mat processed_frame = frameCHW;

    runtime->SetInput(0, processed_frame.ptr<float>());
    runtime->Run(DRPAI_FREQ);

    int32_t i = 0;
    int32_t output_num = 0;
    std::tuple<InOutDataType, void*, int64_t> output_buffer;
    int64_t output_size;
    uint32_t size_count = 0;

    output_num = runtime->GetNumOutput();

    size_count = 0;

    /*GetOutput loop*/
    for (i = 0; i < output_num; i++) {
        /* output_buffer below is tuple, which is { data type, address of output
         * data, number of elements } */
        output_buffer = runtime->GetOutput(i);

        /*Output Data Size = std::get<2>(output_buffer). */
        output_size = std::get<2>(output_buffer);

        /*Output Data Type = std::get<0>(output_buffer)*/
        if (InOutDataType::FLOAT16 == std::get<0>(output_buffer)) {
            uint16_t* data_ptr = reinterpret_cast<uint16_t*>(std::get<1>(output_buffer));
            for (int j = 0; j < output_size; j++) {
                /*FP16 to FP32 conversion*/
                drpai_output_buf[j + size_count] = float16_to_float32(data_ptr[j]);
            }
        } else if (InOutDataType::FLOAT32 == std::get<0>(output_buffer)) {
            /*Output Data = std::get<1>(output_buffer)*/
            float* data_ptr = reinterpret_cast<float*>(std::get<1>(output_buffer));
            for (int j = 0; j < output_size; j++) {
                drpai_output_buf[j + size_count] = data_ptr[j];
            }
        } else {
            LOGR_ERROR("Unsupported output data type.");
            ret = -1;
            break;
        }
        size_count += output_size;
    }

    if (ret != 0) {
        LOGR_ERROR("DRP AI Inference Failed.");
    }

    R_Post_Proc(drpai_output_buf);

    return std::make_tuple(det);
}
#endif