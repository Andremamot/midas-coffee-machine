#ifdef V2H
#pragma once

#include <cstdint>
#include <string>
#include <vector>

/*DRP-AI Input image information*/
#define IMAGE_WIDTH (640)
#define IMAGE_HEIGHT (480)
#define DRPAI_IN_WIDTH (IMAGE_WIDTH)
#define DRPAI_IN_HEIGHT (IMAGE_HEIGHT)
#define BGR_CHANNEL (3)
#define BGRA_CHANNEL (4)
#define DISP_OUTPUT_WIDTH (1920)
#define DISP_OUTPUT_HEIGHT (1080)
#define DISP_INF_WIDTH (1280)
#define DISP_INF_HEIGHT (960)

#define DRPAI_FREQ (3)
/* DRPAI_FREQ can be set from 1 to 127   */
/* 1,2: 1GHz                             */
/* 3: 630MHz                             */
/* 4: 420MHz                             */
/* 5: 315MHz                             */
/* ...                                   */
/* 127: 10MHz                            */
/* Calculation Formula:                  */
/*     1260MHz /(DRPAI_FREQ - 1)         */
/*     (When DRPAI_FREQ = 3 or more.)    */
#endif