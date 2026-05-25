/*******************************************************************************
 * define_midas.h
 * Constants for MiDaS depth estimation via DRP-AI (Renesas RZ/V2H)
 * Adapted from app_midas_cam/src/define.h
 ******************************************************************************/
#pragma once

#ifdef V2H

#include <string>

/*---------------------------------------
 * MiDaS model dimensions
 *--------------------------------------*/
#define MIDAS_MODEL_IN_W    (256)
#define MIDAS_MODEL_IN_H    (256)
#define MIDAS_MODEL_OUT_W   (256)
#define MIDAS_MODEL_OUT_H   (256)
#define MIDAS_NUM_CLASS     (1)

/*---------------------------------------
 * DRP-AI memory offset for MiDaS model
 * (must not overlap with cup model @ 0x0)
 *--------------------------------------*/
#define MIDAS_DRPAI_MEM_OFFSET  (0x8000000)

/*---------------------------------------
 * Model paths (relative to working dir)
 *--------------------------------------*/
namespace c {
class midas {
public:
    inline static const std::string model_dir = "model/midas";
    inline static const std::string pre_dir   = model_dir + "/preprocess";
};
}  // namespace c

#endif  // V2H
