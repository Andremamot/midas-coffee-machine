#ifdef V2H
/***********************************************************************************************************************
 * Copyright (C) 2023 Renesas Electronics Corporation. All rights reserved.
 ***********************************************************************************************************************/
/***********************************************************************************************************************
 * File Name    : dfl_proc.cpp
 * Version      : 2.5.0
 * Description  : RZ/V2H DRP-AI Sample Application for Ultralytics Detection YOLOv8 with MIPI/USB
 * Camera
 ***********************************************************************************************************************/

/*****************************************
 * Includes
 ******************************************/
#include <detections/cup/dfl_proc.h>

#include <opencv2/opencv.hpp>
#include <thread>

DFL::DFL() {}

DFL::~DFL() {}

/*****************************************
 * Function Name : sigmoid
 * Description   : Helper function for YOLO Post Processing
 * Arguments     : x = input argument for the calculation
 * Return value  : sigmoid result of input x
 ******************************************/
double DFL::sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

/*****************************************
 * Function Name : softmax
 * Description   : Helper function for YOLO Post Processing
 * Arguments     : input = input array
 *                 size = size of input array
 * Return value  : softmax result of input array
 ******************************************/
void DFL::softmax(const float* input, int size, float* output) {
    float max_val = *std::max_element(input, input + size);
    float sum = 0.0;

    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

/*****************************************
 * Function Name : stage_conv
 * Description   : Helper function for YOLO Post Processing
 * Arguments     : input = input array
 *                 size = size of input array
 * Return value  : result of convolution
 ******************************************/
float DFL::stage_conv(const float* input, int size) {
    float weight[16];
    for (int i = 0; i < 16; i++) {
        weight[i] = i;
    }

    float result = 0.0;
    for (int i = 0; i < size; i++) {
        result += input[i] * weight[i];
    }

    return result;
}

/*****************************************
 * Function Name : stage_add_0
 * Description   : Helper function for YOLO Post Processing
 * Arguments     : arr = input array
 *                 h, w = array shape
 * Return value  : stage add result of input array
 ******************************************/
float* DFL::stage_add_0(float* arr, int32_t h, int32_t w) {
    float param[w];
    for (int i = 0; i < w; i++) {
        param[i] = i + 0.5;  // [0.5, 1.5, ... , ]
    }

    float* result = new float[h * w];
    for (int _h = 0; _h < h; _h++) {
        for (int _w = 0; _w < w; _w++) {
            result[_h * w + _w] = arr[_h * w + _w] + param[_w];
        }
    }
    return result;
}

/*****************************************
 * Function Name : stage_add_1
 * Description   : Helper function for YOLO Post Processing
 * Arguments     : arr = input array
 *                 h, w = array shape
 * Return value  : stage add result of input array
 ******************************************/
float* DFL::stage_add_1(float* arr, int32_t h, int32_t w) {
    float param = 0.5;
    float* result = new float[h * w];
    for (int _h = 0; _h < h; _h++) {
        for (int _w = 0; _w < w; _w++) {
            result[_h * w + _w] = arr[_h * w + _w] + param;
        }
        param += 1;
    }
    return result;
}

/*****************************************
 * Function Name : stage_sub_0
 * Description   : Helper function for YOLO Post Processing
 * Arguments     : arr = input array
 *                 h, w = array shape
 * Return value  : stage sub result of input array
 ******************************************/
float* DFL::stage_sub_0(float* arr, int32_t h, int32_t w) {
    float param[w];
    for (int i = 0; i < w; i++) {
        param[i] = i + 0.5;  // [0.5, 1.5, ... , ]
    }

    float* result = new float[h * w];
    for (int _h = 0; _h < h; _h++) {
        for (int _w = 0; _w < w; _w++) {
            result[_h * w + _w] = param[_w] - arr[_h * w + _w];
        }
    }
    return result;
}

/*****************************************
 * Function Name : stage_sub_1
 * Description   : Helper function for YOLO Post Processing
 * Arguments     : arr = input array
 *                 h, w = array shape
 * Return value  : stage sub result of input array
 ******************************************/
float* DFL::stage_sub_1(float* arr, int32_t h, int32_t w) {
    float param = 0.5;
    float* result = new float[h * w];
    for (int _h = 0; _h < h; _h++) {
        for (int _w = 0; _w < w; _w++) {
            result[_h * w + _w] = param - arr[_h * w + _w];
        }
        param += 1;
    }
    return result;
}

/*****************************************
 * Function Name : split_dfl
 * Description   : Helper function for YOLO Post Processing
 * Arguments     : dfl_arr = input array
 *                 arr_size = size of input array
 * Return value  : split result of input array
 ******************************************/
float* DFL::split_dfl(float* dfl_arr, int32_t arr_size) {
    int32_t c = 64;
    int32_t h = 0;
    int32_t w = 0;

    if (arr_size == c::yolov8::num_dfl80_out) {
        h = 80;
        w = 80;
    } else if (arr_size == c::yolov8::num_dfl40_out) {
        h = 40;
        w = 40;
    } else if (arr_size == c::yolov8::num_dfl20_out) {
        h = 20;
        w = 20;
    }

    int32_t ch_unit = 16;
    int32_t c_div = c / ch_unit;

    /* Transpose (CHW to HWC) */
    float* dfl_arr_t = new float[h * w * c];
    for (int _h = 0; _h < h; _h++) {
        for (int _w = 0; _w < w; _w++) {
            for (int _c = 0; _c < c; _c++) {
                dfl_arr_t[_h * w * c + _w * c + _c] = dfl_arr[_c * h * w + _h * w + _w];
            }
        }
    }

    /* Softmax operation */
    float* intm_1 = new float[h * w * c];
    for (int _h = 0; _h < h; _h++) {
        for (int _w = 0; _w < w; _w++) {
            for (int _c = 0; _c < c_div; _c++) {
                int idx = _c * ch_unit;
                const float* input_ptr = dfl_arr_t + (_h * w * c + _w * c + idx);
                float* output_ptr = intm_1 + (_h * w * c + _w * c + idx);
                softmax(input_ptr, ch_unit, output_ptr);
            }
        }
    }
    delete[] dfl_arr_t;

    /* "Stage" conv */
    float* intm_2 = new float[h * w * c_div];
    for (int _h = 0; _h < h; _h++) {
        for (int _w = 0; _w < w; _w++) {
            for (int _c = 0; _c < c_div; _c++) {
                int idx = _c * ch_unit;
                const float* input_ptr = intm_1 + (_h * w * c + _w * c + idx);
                float _sc_out = stage_conv(input_ptr, ch_unit);
                intm_2[_h * w * c_div + _w * c_div + _c] = _sc_out;
            }
        }
    }
    delete[] intm_1;

    /* Split data */
    float* intm_2_0 = new float[h * w];
    float* intm_2_1 = new float[h * w];
    float* intm_2_2 = new float[h * w];
    float* intm_2_3 = new float[h * w];
    for (int _h = 0; _h < h; _h++) {
        for (int _w = 0; _w < w; _w++) {
            intm_2_0[_h * w + _w] = intm_2[_h * w * c_div + _w * c_div + 0];
            intm_2_1[_h * w + _w] = intm_2[_h * w * c_div + _w * c_div + 1];
            intm_2_2[_h * w + _w] = intm_2[_h * w * c_div + _w * c_div + 2];
            intm_2_3[_h * w + _w] = intm_2[_h * w * c_div + _w * c_div + 3];
        }
    }
    delete[] intm_2;

    /* Add & Sub by fixed parameter */
    float* intm_3_sub_0 = stage_sub_0(intm_2_0, h, w);
    float* intm_3_sub_1 = stage_sub_1(intm_2_1, h, w);
    float* intm_3_add_0 = stage_add_0(intm_2_2, h, w);
    float* intm_3_add_1 = stage_add_1(intm_2_3, h, w);
    delete[] intm_2_0;
    delete[] intm_2_1;
    delete[] intm_2_2;
    delete[] intm_2_3;

    /* Add & Sub */
    float* intm_4_add_out_0 = new float[h * w];
    for (int i = 0; i < h * w; i++) {
        intm_4_add_out_0[i] = intm_3_sub_0[i] + intm_3_add_0[i];
    }
    float* intm_4_add_out_1 = new float[h * w];
    for (int i = 0; i < h * w; i++) {
        intm_4_add_out_1[i] = intm_3_sub_1[i] + intm_3_add_1[i];
    }
    float* intm_4_sub_out_0 = new float[h * w];
    for (int i = 0; i < h * w; i++) {
        intm_4_sub_out_0[i] = intm_3_add_0[i] - intm_3_sub_0[i];
    }
    float* intm_4_sub_out_1 = new float[h * w];
    for (int i = 0; i < h * w; i++) {
        intm_4_sub_out_1[i] = intm_3_add_1[i] - intm_3_sub_1[i];
    }
    delete[] intm_3_sub_0;
    delete[] intm_3_sub_1;
    delete[] intm_3_add_0;
    delete[] intm_3_add_1;

    /* Div & Mul */
    int32_t div_val = 2;
    int32_t mul_val = 0;
    if (w == 20) {
        mul_val = 32;
    } else if (w == 40) {
        mul_val = 16;
    } else if (w == 80) {
        mul_val = 8;
    }

    float* intm_5_add_0 = new float[h * w];
    for (int i = 0; i < h * w; i++) {
        intm_5_add_0[i] = (intm_4_add_out_0[i] / div_val) * mul_val;
    }
    float* intm_5_add_1 = new float[h * w];
    for (int i = 0; i < h * w; i++) {
        intm_5_add_1[i] = (intm_4_add_out_1[i] / div_val) * mul_val;
    }
    float* intm_5_sub_0 = new float[h * w];
    for (int i = 0; i < h * w; i++) {
        intm_5_sub_0[i] = (intm_4_sub_out_0[i] / 1) * mul_val;
    }
    float* intm_5_sub_1 = new float[h * w];
    for (int i = 0; i < h * w; i++) {
        intm_5_sub_1[i] = (intm_4_sub_out_1[i] / 1) * mul_val;
    }
    delete[] intm_4_add_out_0;
    delete[] intm_4_add_out_1;
    delete[] intm_4_sub_out_0;
    delete[] intm_4_sub_out_1;

    /* Concat */
    int32_t final_channel = 4;
    float* out = new float[h * w * final_channel];
    for (int _h = 0; _h < h; _h++) {
        for (int _w = 0; _w < w; ++_w) {
            int idx = _h * w + _w;
            out[idx * final_channel + 0] = intm_5_add_0[idx];
            out[idx * final_channel + 1] = intm_5_add_1[idx];
            out[idx * final_channel + 2] = intm_5_sub_0[idx];
            out[idx * final_channel + 3] = intm_5_sub_1[idx];
        }
    }
    delete[] intm_5_add_0;
    delete[] intm_5_add_1;
    delete[] intm_5_sub_0;
    delete[] intm_5_sub_1;

    /* HWC to CHW */
    float* final_output = new float[final_channel * h * w];
    for (int c = 0; c < final_channel; c++) {
        for (int _h = 0; _h < h; _h++) {
            for (int _w = 0; _w < w; ++_w) {
                final_output[c * h * w + _h * w + _w] =
                    out[_h * w * final_channel + _w * final_channel + c];
            }
        }
    }
    delete[] out;

    return final_output;
}

/*****************************************
 * Function Name : dfl_process
 * Description   : process for thread
 * Arguments     : -
 * Return value  : -
 ******************************************/
void DFL::dfl_process(float* dfl, uint32_t dfl_size, float* dfl_out) {
    float* tmp = split_dfl(dfl, dfl_size);
    std::copy(tmp, tmp + dfl_size / c::yolov8::REG_MAX, dfl_out);
    delete[] tmp;
}

/*****************************************
 * Function Name : sigmoid_process
 * Description   : process for thread
 * Arguments     : -
 * Return value  : -
 ******************************************/
void DFL::sigmoid_process(float* cls, uint32_t sigmoid_size, float* sigmoid_out) {
#if (1) <= CPU_DFL_SIGMOID_SKIP
    std::copy(cls, cls + sigmoid_size, sigmoid_out);
#else
    for (uint32_t i = 0; i < sigmoid_size; i++) {
        sigmoid_out[i] = sigmoid(cls[i]);
    }
#endif
}

/*****************************************
 * Function Name : DFL_Proc
 * Description   : DFL process for Yolov8
 * Arguments     : dfl80, dfl40, dfl20 = dfl array
 *                 class80, class40, class20 = class array
 * Return value  : -
 ******************************************/
void DFL::DFL_Proc(float* dfl80,
                   float* dfl40,
                   float* dfl20,
                   float* class80,
                   float* class40,
                   float* class20,
                   float* output_buf) {
    int32_t dfl_c = 4;

    /* DFL and Sigmoid operation */
    float* dfl_out_80 = new float[c::yolov8::num_dfl80_out / c::yolov8::REG_MAX];
    float* sigmoid_out_80 = new float[c::yolov8::num_class80_out];
    float* dfl_out_40 = new float[c::yolov8::num_dfl40_out / c::yolov8::REG_MAX];
    float* sigmoid_out_40 = new float[c::yolov8::num_class40_out];
    float* dfl_out_20 = new float[c::yolov8::num_dfl20_out / c::yolov8::REG_MAX];
    float* sigmoid_out_20 = new float[c::yolov8::num_class20_out];

#if (1) == CPU_DFL_MULTI_THREAD
    std::thread thread_dfl_80(&DFL::dfl_process, this, dfl80, c::yolov8::num_dfl80_out, dfl_out_80);
    std::thread thread_sigmoid_80(&DFL::sigmoid_process, this, class80, c::yolov8::num_class80_out,
                                  sigmoid_out_80);
    std::thread thread_dfl_40(&DFL::dfl_process, this, dfl40, c::yolov8::num_dfl40_out, dfl_out_40);
    std::thread thread_sigmoid_40(&DFL::sigmoid_process, this, class40, c::yolov8::num_class40_out,
                                  sigmoid_out_40);
    std::thread thread_dfl_20(&DFL::dfl_process, this, dfl20, c::yolov8::num_dfl20_out, dfl_out_20);
    std::thread thread_sigmoid_20(&DFL::sigmoid_process, this, class20, c::yolov8::num_class20_out,
                                  sigmoid_out_20);
    thread_dfl_80.join();
    thread_sigmoid_80.join();
    thread_dfl_40.join();
    thread_sigmoid_40.join();
    thread_dfl_20.join();
    thread_sigmoid_20.join();
#else
    DFL::dfl_process(dfl80, num_dfl80_out, dfl_out_80);
    DFL::dfl_process(dfl40, num_dfl40_out, dfl_out_40);
    DFL::dfl_process(dfl20, num_dfl20_out, dfl_out_20);
    DFL::sigmoid_process(class80, num_class80_out, sigmoid_out_80);
    DFL::sigmoid_process(class40, num_class40_out, sigmoid_out_40);
    DFL::sigmoid_process(class20, num_class20_out, sigmoid_out_20);
#endif

    /* Reshape & Concat operation */
    /* Reshape dfl out (4, 8400) */
    float* dfl_all_out = new float[dfl_c * c::yolov8::num_grid_points];
    for (int c = 0; c < dfl_c; c++) {
        int32_t idx = 0;

        /* dfl_out_80 */
        for (int i = 0; i < c::yolov8::num_grids[0]; i++) {
            for (int j = 0; j < c::yolov8::num_grids[0]; j++) {
                dfl_all_out[c * c::yolov8::num_grid_points + idx] =
                    dfl_out_80[c * c::yolov8::num_grids[0] * c::yolov8::num_grids[0] +
                               i * c::yolov8::num_grids[0] + j];
                idx++;
            }
        }
        /* dfl_out_40 */
        for (int i = 0; i < c::yolov8::num_grids[1]; i++) {
            for (int j = 0; j < c::yolov8::num_grids[1]; j++) {
                dfl_all_out[c * c::yolov8::num_grid_points + idx] =
                    dfl_out_40[c * c::yolov8::num_grids[1] * c::yolov8::num_grids[1] +
                               i * c::yolov8::num_grids[1] + j];
                idx++;
            }
        }
        /* dfl_out_20 */
        for (int i = 0; i < c::yolov8::num_grids[2]; i++) {
            for (int j = 0; j < c::yolov8::num_grids[2]; j++) {
                dfl_all_out[c * c::yolov8::num_grid_points + idx] =
                    dfl_out_20[c * c::yolov8::num_grids[2] * c::yolov8::num_grids[2] +
                               i * c::yolov8::num_grids[2] + j];
                idx++;
            }
        }
    }
    delete[] dfl_out_80;
    delete[] dfl_out_40;
    delete[] dfl_out_20;

    /* Reshape sigmoid out (80, 8400) */
    float* sigmoid_all_out = new float[c::yolov8::NUM_CLASS * c::yolov8::num_grid_points];
    for (int c = 0; c < c::yolov8::NUM_CLASS; c++) {
        int32_t idx = 0;

        /* sigmoid_out_80 */
        for (int i = 0; i < c::yolov8::num_grids[0]; i++) {
            for (int j = 0; j < c::yolov8::num_grids[0]; j++) {
                sigmoid_all_out[c * c::yolov8::num_grid_points + idx] =
                    sigmoid_out_80[c * c::yolov8::num_grids[0] * c::yolov8::num_grids[0] +
                                   i * c::yolov8::num_grids[0] + j];
                idx++;
            }
        }
        /* sigmoid_out_40 */
        for (int i = 0; i < c::yolov8::num_grids[1]; i++) {
            for (int j = 0; j < c::yolov8::num_grids[1]; j++) {
                sigmoid_all_out[c * c::yolov8::num_grid_points + idx] =
                    sigmoid_out_40[c * c::yolov8::num_grids[1] * c::yolov8::num_grids[1] +
                                   i * c::yolov8::num_grids[1] + j];
                idx++;
            }
        }
        /* sigmoid_out_20 */
        for (int i = 0; i < c::yolov8::num_grids[2]; i++) {
            for (int j = 0; j < c::yolov8::num_grids[2]; j++) {
                sigmoid_all_out[c * c::yolov8::num_grid_points + idx] =
                    sigmoid_out_20[c * c::yolov8::num_grids[2] * c::yolov8::num_grids[2] +
                                   i * c::yolov8::num_grids[2] + j];
                idx++;
            }
        }
    }
    delete[] sigmoid_out_80;
    delete[] sigmoid_out_40;
    delete[] sigmoid_out_20;

    /* Concat (84, 8400) */
    /* dfl_all_out */
    for (int i = 0; i < dfl_c; i++) {
        for (uint32_t j = 0; j < c::yolov8::num_grid_points; j++) {
            output_buf[i * c::yolov8::num_grid_points + j] =
                dfl_all_out[i * c::yolov8::num_grid_points + j];
        }
    }
    /* sigmoid_all_out */
    for (int i = 0; i < c::yolov8::NUM_CLASS; i++) {
        for (uint32_t j = 0; j < c::yolov8::num_grid_points; j++) {
            output_buf[(dfl_c + i) * c::yolov8::num_grid_points + j] =
                sigmoid_all_out[i * c::yolov8::num_grid_points + j];
        }
    }
    delete[] dfl_all_out;
    delete[] sigmoid_all_out;

    return;
}
#endif