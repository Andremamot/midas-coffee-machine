
/***********************************************************************************************************************
 * DISCLAIMER
 * This software is supplied by Renesas Electronics Corporation and is only
 * intended for use with Renesas products. No other uses are authorized. This
 * software is owned by Renesas Electronics Corporation and is protected under
 * all applicable laws, including copyright laws. THIS SOFTWARE IS PROVIDED "AS
 * IS" AND RENESAS MAKES NO WARRANTIES REGARDING THIS SOFTWARE, WHETHER EXPRESS,
 * IMPLIED OR STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. ALL
 * SUCH WARRANTIES ARE EXPRESSLY DISCLAIMED. TO THE MAXIMUM EXTENT PERMITTED NOT
 * PROHIBITED BY LAW, NEITHER RENESAS ELECTRONICS CORPORATION NOR ANY OF ITS
 * AFFILIATED COMPANIES SHALL BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL,
 * INCIDENTAL OR CONSEQUENTIAL DAMAGES FOR ANY REASON RELATED TO THIS SOFTWARE,
 * EVEN IF RENESAS OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES. Renesas reserves the right, without notice, to make changes to
 * this software and to discontinue the availability of this software. By using
 * this software, you agree to the additional terms and conditions found by
 * accessing the following link: http://www.renesas.com/disclaimer
 *
 * Copyright (C) 2023 Renesas Electronics Corporation. All rights reserved.
 ***********************************************************************************************************************/
/***********************************************************************************************************************
 * File Name    : box.h
 * Version      : 1.1.1
 * Description  : RZ/V2MA DRP-AI TVM[*1] Sample Application for USB Camera HTTP
 * version *1 DRP-AI TVM is powered by EdgeCortix MERA(TM) Compiler Framework.
 ***********************************************************************************************************************/

#pragma once

#include <math.h>
#include <stdlib.h>

#include <vector>

/*****************************************
 * Box : Bounding box coordinates and its size
 ******************************************/
struct Box {
    float x;  // top-left x
    float y;  // top-left y
    float w;
    float h;

    float area() const { return w * h; }

    Box intersect(const Box& other) const {
        float x1 = std::max(x, other.x);
        float y1 = std::max(y, other.y);
        float x2 = std::min(x + w, other.x + other.w);
        float y2 = std::min(y + h, other.y + other.h);

        float iw = std::max(0.0f, x2 - x1);
        float ih = std::max(0.0f, y2 - y1);

        return {x1, y1, iw, ih};
    }

    bool contains(float px, float py) const {
        return (px >= x && px <= x + w && py >= y && py <= y + h);
    }
};

/*****************************************
 * detection : Detected result
 ******************************************/
struct Detection {
    Box box;
    float score;
    int class_id;
};

/*****************************************
 * Functions
 ******************************************/
float box_iou(Box a, Box b);
float overlap(float x1, float w1, float x2, float w2);
float box_intersection(Box a, Box b);
float box_union(Box a, Box b);
void filter_boxes_nms(std::vector<Detection>& det, int32_t size, float th_nms);
