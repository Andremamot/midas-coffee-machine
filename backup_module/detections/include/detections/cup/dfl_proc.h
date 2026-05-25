#ifdef V2H
/***********************************************************************************************************************
 * Copyright (C) 2023 Renesas Electronics Corporation. All rights reserved.
 ***********************************************************************************************************************/
/***********************************************************************************************************************
 * File Name    : dfl_proc.h
 * Version      : 2.5.0
 * Description  : RZ/V2H DRP-AI Sample Application for Ultralytics Detection
 * YOLOv8 with MIPI/USB Camera
 ***********************************************************************************************************************/

#pragma once

#include "constants/define_cup.h"

class DFL {
public:
    DFL();
    ~DFL();

    void DFL_Proc(float* dfl80,
                  float* dfl40,
                  float* dfl20,
                  float* class80,
                  float* class40,
                  float* class20,
                  float* output_buf);
    float* split_dfl(float* dfl_arr, int32_t arr_size);
    double sigmoid(double x);

private:
    void softmax(const float* input, int size, float* output);
    float stage_conv(const float* input, int size);
    float* stage_add_0(float* arr, int32_t h, int32_t w);
    float* stage_add_1(float* arr, int32_t h, int32_t w);
    float* stage_sub_0(float* arr, int32_t h, int32_t w);
    float* stage_sub_1(float* arr, int32_t h, int32_t w);
    void dfl_process(float* dfl, uint32_t dfl_size, float* dfl_out);
    void sigmoid_process(float* cls, uint32_t sigmoid_size, float* sigmoid_out);
};

#endif