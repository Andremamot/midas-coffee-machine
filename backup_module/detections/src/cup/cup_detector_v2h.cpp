#if defined(V2H) && (DRP_AI_TVM_RUNTIME == 1)
#include <detections/ai.h>
#include <detections/cup/cup_detector_v2h.h>
#include <detections/drp_queue.h>

CupDetectorV2H::CupDetectorV2H() {
  dfl = DFL();
  drpai_buf = (dma_buffer *)malloc(sizeof(dma_buffer));

  // DRP-AI PreRuntime menerima FORMAT_BGR secara dinamis
  // Ukuran: IMAGE_WIDTH * IMAGE_HEIGHT * BGR_CHANNEL
  int ret =
      buffer_alloc_dmabuf(drpai_buf, IMAGE_WIDTH * IMAGE_HEIGHT * BGR_CHANNEL);
  if (-1 == ret) {
    LOGR_ERROR("Failed to Allocate DMA buffer for the drpai_buf");
    free(drpai_buf);
  }
}

CupDetectorV2H::~CupDetectorV2H() {
  free(drpai_buf);
  drpai_buf = NULL;
}

void CupDetectorV2H::R_Post_Proc(float *floatarr) {
  std::vector<Detection> det_buff;
  uint32_t i = 0;
  uint32_t j = 0;
  float score = 0;
  float probability = 0;
  float center_x = 0;
  float center_y = 0;
  float box_w = 0;
  float box_h = 0;
  int32_t pred_class = -1;
  Detection d;

  float predictions[c::yolov8::num_grid_points][c::yolov8::NUM_CLASS + 4];

  /* Convert 2D array and Transpose */
  for (j = 0; j < c::yolov8::NUM_CLASS + 4; j++) {
    for (i = 0; i < c::yolov8::num_grid_points; i++) {
      predictions[i][j] = floatarr[j * c::yolov8::num_grid_points + i];
    }
  }

#if (2) <= CPU_DFL_SIGMOID_SKIP
  /* Threshold for non-sigmoid value */
  float th_prob = 0;
  if (c::yolov8::TH_PROB <= 0) {
    th_prob = -FLT_MAX;
  } else if (c::yolov8::TH_PROB >= 1) {
    th_prob = FLT_MAX;
  } else {
    th_prob = logf(c::yolov8::TH_PROB / (1.0f - c::yolov8::TH_PROB));
  }
  printf("TH_PROB = %f\n", c::yolov8::TH_PROB);
  printf("th_prob = %f\n", th_prob);
  printf("sigmoid(th_prob) = %f\n", dfl.sigmoid(th_prob));
#else
  /* Threshold for sigmoid value */
  float th_prob = c::yolov8::TH_PROB;
#endif

  for (i = 0; i < c::yolov8::num_grid_points; i++) {
#if (1) <= CPU_DFL_SIGMOID_SKIP
    float max_pred = -FLT_MAX;
#else
    float max_pred = 0;
#endif
    for (j = 0; j < c::yolov8::NUM_CLASS; j++) {
      score = predictions[i][j + 4];
      if (score > max_pred) {
        pred_class = j;
        max_pred = score;
      }
    }

    /* Store the result into the list if the probability is more than the
     * threshold */
    probability = max_pred;
#if (1) == CPU_DFL_SIGMOID_SKIP
    probability = dfl.sigmoid(probability);
#endif
    if (probability > th_prob) {
      /* Adjustment for size */
      /* correct_yolo/region_boxes */
      center_x = predictions[i][0];
      center_y = predictions[i][1];
      box_w = predictions[i][2];
      box_h = predictions[i][3];

      printf("cx=%f cy=%f w=%f h=%f cls0=%f cls1=%f\n", predictions[i][0],
             predictions[i][1], predictions[i][2], predictions[i][3],
             predictions[i][4], predictions[i][5]);
#if (2) <= CPU_DFL_SIGMOID_SKIP
      probability = dfl.sigmoid(probability);
#endif
      // Box bb = {center_x, center_y, box_w, box_h};
      Box bb;
      bb.x = center_x - box_w / 2;
      bb.y = center_y - box_h / 2;
      bb.w = box_w;
      bb.h = box_h;
      d.box = bb;
      d.class_id = pred_class;
      d.score = probability;
      det_buff.push_back(d);
    }
  }

  /* Non-Maximum Supression filter */
  filter_boxes_nms(det_buff, det_buff.size(), c::yolov8::TH_NMS);

  /* Log Output */
  int iBoxCount = 0;
  for (i = 0; i < det_buff.size(); i++) {
    /* Skip the overlapped bounding boxes */
    if (det_buff[i].score == 0)
      continue;

    /* Revert letterbox padding DRP-AI.
     * Model expect 640x640, input is 640x480. PreRuntime menambahkan padding Y
     * = 80 di atas dan bawah. Jadi koordinat output perlu dikurangi 80 di sumbu
     * Y. */
    float pad_y =
        (float)(c::yolov8::MODEL_IN_H - DRPAI_IN_HEIGHT) / 2.0f; // 80.0
    float pad_x = (float)(c::yolov8::MODEL_IN_W - DRPAI_IN_WIDTH) / 2.0f; // 0.0

    det_buff[i].box.x -= pad_x;
    det_buff[i].box.y -= pad_y;
    // Lebar dan tinggi tetap sama karena aspect ratio dipertahankan
  }

  /* Scale detection boxes dari MODEL coords ke resolusi frame asli (DRPAI_IN =
   * 640x480 → original frame) Bounding box dari R_Post_Proc sudah di-scale ke
   * DRPAI_IN_WIDTH/HEIGHT. Kita perlu scale lagi ke resolusi frame asli yang
   * dikirim user. */
  det.clear();
  copy(det_buff.begin(), det_buff.end(), back_inserter(det));
  return;
}

float CupDetectorV2H::float16_to_float32(uint16_t a) {
  return __extendXfYf2__<uint16_t, uint16_t, 10, float, uint32_t, 23>(a);
}

int8_t CupDetectorV2H::get_result() {
  int8_t ret = 0;
  int32_t i = 0;
  int32_t output_num = 0;
  std::tuple<InOutDataType, void *, int64_t> output_buffer;
  int64_t output_size;

  /* Get the number of output of the target model. */
  output_num = runtime->GetNumOutput();
  /*GetOutput loop*/
  for (i = 0; i < output_num; i++) {
    /* output_buffer below is tuple, which is { data type, address of output
     * data, number of elements } */
    output_buffer = runtime->GetOutput(i);
    /*Output Data Size = std::get<2>(output_buffer). */
    output_size = std::get<2>(output_buffer);

    /*Output Data Type = std::get<0>(output_buffer)*/
    if (InOutDataType::FLOAT16 == std::get<0>(output_buffer)) {
      /*Output Data = std::get<1>(output_buffer)*/
      uint16_t *data_ptr =
          reinterpret_cast<uint16_t *>(std::get<1>(output_buffer));

      for (int j = 0; j < output_size; j++) {
        /*FP16 to FP32 conversion*/
        switch (output_size) {
        case c::yolov8::num_dfl80_out:
          c::yolov8::output_dfl80[j] = float16_to_float32(data_ptr[j]);
          break;
        case c::yolov8::num_dfl40_out:
          c::yolov8::output_dfl40[j] = float16_to_float32(data_ptr[j]);
          break;
        case c::yolov8::num_dfl20_out:
          c::yolov8::output_dfl20[j] = float16_to_float32(data_ptr[j]);
          break;
        case c::yolov8::num_class80_out:
          c::yolov8::output_class80[j] = float16_to_float32(data_ptr[j]);
          break;
        case c::yolov8::num_class40_out:
          c::yolov8::output_class40[j] = float16_to_float32(data_ptr[j]);
          break;
        case c::yolov8::num_class20_out:
          c::yolov8::output_class20[j] = float16_to_float32(data_ptr[j]);
          break;
        default:
          break;
        }
      }
    } else if (InOutDataType::FLOAT32 == std::get<0>(output_buffer)) {
      /*Output Data = std::get<1>(output_buffer)*/
      float *data_ptr = reinterpret_cast<float *>(std::get<1>(output_buffer));
      for (int j = 0; j < output_size; j++) {
        switch (output_size) {
        case c::yolov8::num_dfl80_out:
          c::yolov8::output_dfl80[j] = data_ptr[j];
          break;
        case c::yolov8::num_dfl40_out:
          c::yolov8::output_dfl40[j] = data_ptr[j];
          break;
        case c::yolov8::num_dfl20_out:
          c::yolov8::output_dfl20[j] = data_ptr[j];
          break;
        case c::yolov8::num_class80_out:
          c::yolov8::output_class80[j] = data_ptr[j];
          break;
        case c::yolov8::num_class40_out:
          c::yolov8::output_class40[j] = data_ptr[j];
          break;
        case c::yolov8::num_class20_out:
          c::yolov8::output_class20[j] = data_ptr[j];
          break;
        default:
          break;
        }
      }
    } else {
      fprintf(stderr, "[ERROR] Output data type : not floating point.\n");
      ret = -1;
      break;
    }
  }
  return ret;
}

std::tuple<std::vector<Detection>> CupDetectorV2H::detect(cv::Mat &frame) {
  auto future = DRPQueue::get_instance().enqueue([this, &frame]() {
    void *framePtr;
    uint32_t out_size;
    int8_t ret = 0;

    /* ── Preprocess: resize frame BGR ke resolusi DRP-AI ───────────────── */
    cv::Mat frame_for_drp;

    /* 1. Resize ke resolusi input DRP-AI (640x480) jika perlu */
    if (frame.cols != IMAGE_WIDTH || frame.rows != IMAGE_HEIGHT) {
      cv::resize(frame, frame_for_drp, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), 0,
                 0, cv::INTER_LINEAR);
    } else {
      frame_for_drp = frame;
    }

    /* Pastikan format BGR */
    if (frame_for_drp.type() != CV_8UC3) {
      cv::cvtColor(frame_for_drp, frame_for_drp, cv::COLOR_GRAY2BGR);
    }

    /* 3. Salin ke DMA buffer */
    size_t bgr_size = (size_t)IMAGE_WIDTH * IMAGE_HEIGHT * BGR_CHANNEL;
    memcpy(drpai_buf->mem, frame_for_drp.data, bgr_size);
    ret = buffer_flush_dmabuf(drpai_buf->idx, bgr_size);

    if (ret < 0) {
      LOGR_ERROR("CupDetectorV2H: Buffer Flush Failed!");
      return std::make_tuple(std::vector<Detection>());
    }

    in_param.pre_in_shape_w = IMAGE_WIDTH;
    in_param.pre_in_shape_h = IMAGE_HEIGHT;
    in_param.pre_in_format = FORMAT_BGR;
    in_param.pre_out_format = FORMAT_RGB;

    /* Reload YOLOv8 Pre-processing efficiently via AI singleton cache */
    AI::get_instance()->ReloadPre(c::yolov8::pre_dir);

    float mean[] = {0, 0, 0};
    float std[] = {1, 1, 1};
    in_param.cof_add[0] = -255 * mean[0];
    in_param.cof_add[1] = -255 * mean[1];
    in_param.cof_add[2] = -255 * mean[2];
    in_param.cof_mul[0] = 1 / (std[0] * 255);
    in_param.cof_mul[1] = 1 / (std[1] * 255);
    in_param.cof_mul[2] = 1 / (std[2] * 255);

    in_param.pre_in_addr = (uintptr_t)drpai_buf->phy_addr;
    ret = preruntime->Pre(&in_param, &framePtr, &out_size);

    if (ret < 0) {
      LOGR_ERROR("CupDetectorV2H: Pre Process Failed!");
      return std::make_tuple(std::vector<Detection>());
    }

    runtime->SetInput(0, (float *)framePtr);
    runtime->Run(DRPAI_FREQ);

    ret = get_result();
    if (ret < 0) {
      LOGR_ERROR("CupDetectorV2H: Get Result Failed!");
      return std::make_tuple(std::vector<Detection>());
    }

    dfl.DFL_Proc(c::yolov8::output_dfl80, c::yolov8::output_dfl40,
                 c::yolov8::output_dfl20, c::yolov8::output_class80,
                 c::yolov8::output_class40, c::yolov8::output_class20,
                 drpai_output_buf);

    R_Post_Proc(drpai_output_buf);

    /* Scale bbox dari DRPAI_IN (640×480) ke resolusi frame asli yang dikirim.
     * R_Post_Proc menghasilkan koordinat dalam ruang DRPAI_IN_WIDTH x
     * DRPAI_IN_HEIGHT. */
    float sx = (float)frame.cols / (float)DRPAI_IN_WIDTH;
    float sy = (float)frame.rows / (float)DRPAI_IN_HEIGHT;
    if (sx != 1.0f || sy != 1.0f) {
      for (auto &d : det) {
        d.box.x *= sx;
        d.box.y *= sy;
        d.box.w *= sx;
        d.box.h *= sy;
      }
    }

    return std::make_tuple(det);
  });
  return future.get();
}

bool CupDetectorV2H::check_cup_in_center(
    const std::vector<Detection> &detections, const cv::Point &center_point) {
  // check is rim in center
  bool is_center_in_rim = false;
  std::vector<Box> cup_rim_boxes;
  std::vector<Box> cup_body_boxes;

  for (const auto &det : detections) {
    if (det.class_id == 0) // "cup" (rim)
      cup_rim_boxes.push_back(det.box);

    else if (det.class_id == 1) // "cup_body"
      cup_body_boxes.push_back(det.box);
  }

  for (const auto &rim_box : cup_rim_boxes) {
    int rim_center_x = rim_box.x + rim_box.w / 2;
    int rim_center_y = rim_box.y + rim_box.h / 2;

    bool associated_body_found = false;

    for (const auto &body_box : cup_body_boxes) {
      if (body_box.contains(rim_center_x, rim_center_y)) {
        associated_body_found = true;
        break;
      }
    }
    if (!associated_body_found)
      continue;

    float margin_x_r = rim_box.w * TOLERANCE_FACTOR;
    float margin_y_r = rim_box.h * TOLERANCE_FACTOR;

    float inner_x1_r = rim_box.x + margin_x_r;
    float inner_y1_r = rim_box.y + margin_y_r;
    float inner_x2_r = rim_box.x + rim_box.w - margin_x_r;
    float inner_y2_r = rim_box.y + rim_box.h - margin_y_r;

    if (center_point.x >= inner_x1_r && center_point.x <= inner_x2_r &&
        center_point.y >= inner_y1_r && center_point.y <= inner_y2_r) {
      is_center_in_rim = true;
      break;
    }
  }
  return is_center_in_rim;
}

// /* new post process*/
// std::vector<Detection> CupDetectorV2H::postprocess(float *cls_8, float
// *cls_16, float *cls_32, float *bb_8, float *bb_16, float *bb_32)
// {
//     std::vector<Detection> dets, new_dets;

//     decode(cls_8, bb_8, 8, dets);
//     decode(cls_16, bb_16, 16, dets);
//     decode(cls_32, bb_32, 32, dets);

//     printf("PreNMS: %d\n", dets.size());
//     // dets = nms(dets, thresh_nms);
//     dets = hard_nms(dets, c::yolov8::TH_NMS, -1, 100);

//     for (int i = 0; i < dets.size(); i++)
//     {
//         Detection det = dets[i];
//         float w = det.bbox.x2 - det.bbox.x1;
//         if (w > 1)
//             continue;
//         new_dets.push_back(det);
//     }

//     printf("AfterNMS: %d\n", new_dets.size());
//     for (int i = 0; i < new_dets.size(); i++)
//     {
//         Detection det = new_dets[i];
//         printf("x1: %f, y1: %f, x2: %f, y2: %f, class: %d, prob: %f \n",
//         det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2, det.label,
//         det.class_prob);
//     }
//     return new_dets;
// }

// std::vector<std::vector<float>> CupDetectorV2H::make_anchor(int h, int w,
// float grid_cell_offset)
// {
//     std::vector<std::vector<float>> anchor_points;
//     std::vector<float> sx(w);
//     for (int i = 0; i < w; i++)
//     {
//         sx[i] = static_cast<float>(i) + grid_cell_offset;
//     }
//     std::vector<float> sy(h);
//     for (int i = 0; i < h; i++)
//     {
//         sy[i] = static_cast<float>(i) + grid_cell_offset;
//     }
//     std::vector<std::vector<float>> mesh(h, std::vector<float>(w));
//     for (int i = 0; i < h; i++)
//     {
//         for (int j = 0; j < w; j++)
//         {
//             mesh[i][j] = sy[i];
//         }
//     }
//     for (int j = 0; j < w; j++)
//     {
//         for (int i = 0; i < h; i++)
//         {
//             mesh[i][j] = sx[j];
//         }
//     }
//     std::vector<std::vector<float>> flat(h * w, std::vector<float>(2));
//     for (int i = 0; i < h; i++)
//     {
//         for (int j = 0; j < w; j++)
//         {
//             flat[i * w + j][0] = mesh[i][j];
//             flat[i * w + j][1] = sy[i];
//         }
//     }
//     std::swap(flat[0][0], flat[0][1]);
//     std::swap(flat[1][0], flat[1][1]);
//     return flat;
// }

// std::vector<std::vector<float>>
// CupDetectorV2H::dist2bbox(std::vector<std::vector<float>> distance,
// std::vector<std::vector<float>> anchor_points, bool xywh, int dim)
// {
//     //  lt is  2 first values of each row of distance
//     std::vector<std::vector<float>> lt(distance.size(),
//     std::vector<float>(2)); std::vector<std::vector<float>>
//     rb(distance.size(), std::vector<float>(2)); for (int i = 0; i <
//     distance.size(); i++)
//     {
//         for (int j = 0; j < 2; j++)
//         {
//             lt[i][j] = distance[i][j];
//             rb[i][j] = distance[i][j + 2];
//         }
//     }

//     std::vector<std::vector<float>> x1y1(anchor_points.size(),
//     std::vector<float>(2)); std::vector<std::vector<float>>
//     x2y2(anchor_points.size(), std::vector<float>(2)); for (int i = 0; i <
//     anchor_points.size(); i++)
//     {
//         for (int j = 0; j < 2; j++)
//         {
//             x1y1[i][j] = anchor_points[i][j] - lt[i][j];
//             x2y2[i][j] = anchor_points[i][j] + rb[i][j];
//         }
//     }
//     if (xywh)
//     {
//         std::vector<std::vector<float>> c_xy(anchor_points.size(),
//         std::vector<float>(2)); std::vector<std::vector<float>>
//         wh(anchor_points.size(), std::vector<float>(2)); for (int i = 0; i <
//         anchor_points.size(); i++)
//         {
//             for (int j = 0; j < 2; j++)
//             {
//                 c_xy[i][j] = (x1y1[i][j] + x2y2[i][j]) / 2;
//                 wh[i][j] = x2y2[i][j] - x1y1[i][j];
//             }
//         }
//         std::vector<std::vector<float>> bbox(anchor_points.size(),
//         std::vector<float>(4)); for (int i = 0; i < anchor_points.size();
//         i++)
//         {
//             bbox[i][0] = c_xy[i][0];
//             bbox[i][1] = c_xy[i][1];
//             bbox[i][2] = wh[i][0];
//             bbox[i][3] = wh[i][1];
//         }
//         return bbox;
//     }
//     else
//     {
//         std::vector<std::vector<float>> bbox(anchor_points.size(),
//         std::vector<float>(4)); for (int i = 0; i < anchor_points.size();
//         i++)
//         {
//             bbox[i][0] = x1y1[i][0];
//             bbox[i][1] = x1y1[i][1];
//             bbox[i][2] = x2y2[i][0];
//             bbox[i][3] = x2y2[i][1];
//         }
//         return bbox;
//     }
// }
// std::vector<float> CupDetectorV2H::dist2bbox_target(std::vector<float> bbox,
// std::vector<float> anchor_points, bool xywh, int dim)
// {
//     std::vector<float> lt(2);
//     std::vector<float> rb(2);
//     for (int i = 0; i < 2; i++)
//     {
//         lt[i] = bbox[i];
//         rb[i] = bbox[i + 2];
//     }
//     std::vector<float> x1y1(2);
//     std::vector<float> x2y2(2);
//     for (int i = 0; i < 2; i++)
//     {
//         x1y1[i] = anchor_points[i] - lt[i];
//         x2y2[i] = anchor_points[i] + rb[i];
//     }
//     if (xywh)
//     {
//         std::vector<float> c_xy(2);
//         std::vector<float> wh(2);
//         for (int i = 0; i < 2; i++)
//         {
//             c_xy[i] = (x1y1[i] + x2y2[i]) / 2;
//             wh[i] = x2y2[i] - x1y1[i];
//         }
//         std::vector<float> bbox(4);
//         for (int i = 0; i < 2; i++)
//         {
//             bbox[i] = c_xy[i];
//             bbox[i + 2] = wh[i];
//         }
//         return bbox;
//     }
//     else
//     {
//         std::vector<float> bbox(4);
//         for (int i = 0; i < 2; i++)
//         {
//             bbox[i] = x1y1[i];
//             bbox[i + 2] = x2y2[i];
//         }
//         return bbox;
//     }
// }

// std::vector<std::vector<float>> CupDetectorV2H::dfl(float *bb, int h, int w)
// {
//     // std::vector<float>conv = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
//     int num_bb = h * w;
//     std::vector<std::vector<float>> bboxes(num_bb, std::vector<float>(4));

//     for (int i = 0; i < num_bb; i++)
//     {
//         for (int j = 0; j < 4; j++)
//         {
//             std::vector<float> cal(c::yolov8::REG_MAX);
//             float sum = 0;
//             for (int k = 0; k < c::yolov8::REG_MAX; k++)
//             {
//                 cal[k] = bb[c::yolov8::REG_MAX * 4 * i + c::yolov8::REG_MAX *
//                 j + k];
//             }
//             //  do softmax for cal
//             float sum_exp = 0;
//             for (int k = 0; k < c::yolov8::REG_MAX; k++)
//             {
//                 sum_exp += exp(cal[k]);
//             }
//             for (int k = 0; k < c::yolov8::REG_MAX; k++)
//             {
//                 cal[k] = exp(cal[k]) / sum_exp;
//             }
//             for (int k = 0; k < c::yolov8::REG_MAX; k++)
//             {
//                 sum += (cal[k] * conv[k]);
//             }
//             bboxes[i][j] = sum;
//         }
//     }
//     return bboxes;
// }

// std::vector<float> CupDetectorV2H::dfl_target(float *bb, int target_idx)
// {
//     // std::vector<float>conv = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
//     std::vector<float> bbox(4);
//     for (int j = 0; j < 4; j++)
//     {
//         std::vector<float> cal(c::yolov8::REG_MAX);
//         float sum = 0;
//         for (int k = 0; k < c::yolov8::REG_MAX; k++)
//         {
//             cal[k] = bb[c::yolov8::REG_MAX * 4 * target_idx +
//             c::yolov8::REG_MAX * j + k];
//         }
//         //  do softmax for cal
//         float sum_exp = 0;
//         for (int k = 0; k < c::yolov8::REG_MAX; k++)
//         {
//             sum_exp += exp(cal[k]);
//         }
//         for (int k = 0; k < c::yolov8::REG_MAX; k++)
//         {
//             cal[k] = exp(cal[k]) / sum_exp;
//         }
//         for (int k = 0; k < c::yolov8::REG_MAX; k++)
//         {
//             sum += (cal[k] * conv[k]);
//         }
//         bbox[j] = sum;
//     }
//     return bbox;
// }

// void CupDetectorV2H::decode(float *cls, float *bb, int stride,
// std::vector<Detection> &dets)
// {
//     int h = c::yolov8::MODEL_IN_H / stride;
//     int w = c::yolov8::MODEL_IN_W / stride;
//     std::vector<std::vector<float>> anchors = make_anchor(h, w);

//     int num_bb = h * w;
//     // int num_dets = 0;
//     for (int i = 0; i < num_bb; i++)
//     {
//         float cls_max = 0;
//         int cls_max_idx = 0;
//         for (int j = 0; j < c::yolov8::NUM_CLASS; j++)
//         {
//             if (cls[c::yolov8::NUM_CLASS * i + j] > cls_max)
//             {
//                 cls_max = cls[c::yolov8::NUM_CLASS * i + j];
//                 cls_max_idx = j;
//             }
//         }

//         Detection det;
//         if (cls_max > c::yolov8::TH_NMS)
//         {

//             det.label = cls_max_idx;
//             det.class_prob = cls_max;
//             std::vector<float> bbox_target = dfl_target(bb, i);
//             bbox_target = dist2bbox_target(bbox_target, anchors[i], true, 1);

//             float x = bbox_target[0] * stride / c::yolov8::MODEL_IN_W;
//             float y = bbox_target[1] * stride / c::yolov8::MODEL_IN_H;
//             float w = bbox_target[2] * stride / c::yolov8::MODEL_IN_W;
//             float h = bbox_target[3] * stride / c::yolov8::MODEL_IN_H;

//             det.bbox.x1 = x - w / 2;
//             det.bbox.y1 = y - h / 2;
//             det.bbox.x2 = x + w / 2;
//             det.bbox.y2 = y + h / 2;
//             dets.push_back(det);
//             // num_dets++;
//         }
//     }
//     // std::cout << "num_dets = " << num_dets << std::endl;
// }

// std::vector<Detection> nms(std::vector<Detection> &outputs, float
// iou_threshold)
// {
//     std::vector<Detection> selected_outputs;
//     std::sort(outputs.begin(), outputs.end(), [](const Detection &a, const
//     Detection &b) -> bool
//               { return a.class_prob > b.class_prob; });
//     for (unsigned int i = 0; i < outputs.size(); i++)
//     {
//         bool non_overlap = true;
//         for (unsigned int j = 0; j < selected_outputs.size(); j++)
//         {
//             float iou_score = math_utils::iou(selected_outputs[j].bbox,
//             outputs[i].bbox); if (iou_score > iou_threshold)
//             {
//                 non_overlap = false;
//                 break;
//             }
//         }
//         if (non_overlap)
//             selected_outputs.push_back(outputs[i]);
//     }
//     return selected_outputs;
// }

// std::vector<Detection> hard_nms(std::vector<Detection> &box_scores, float
// iou_threshold, int top_k = -1, int candidate_size = 400)
// {

//     std::vector<Detection> selected_boxes;
//     std::vector<int> picked;
//     std::sort(box_scores.begin(), box_scores.end(), [](const Detection &a,
//     const Detection &b) -> bool
//               { return a.class_prob > b.class_prob; });
//     int size = std::min(static_cast<int>(box_scores.size()), candidate_size);
//     for (int i = 0; i < size; ++i)
//     {
//         if (box_scores[i].class_prob == 0)
//         {
//             continue;
//         }
//         picked.push_back(i);
//         if (top_k > 0 && picked.size() == top_k)
//         {
//             break;
//         }
//         for (int j = i + 1; j < size; ++j)
//         {
//             if (box_scores[j].class_prob == 0)
//             {
//                 continue;
//             }
//             if (math_utils::iou(box_scores[picked.back()].bbox,
//             box_scores[j].bbox) > iou_threshold)
//             {
//                 box_scores[j].class_prob = 0;
//             }
//         }
//         std::sort(box_scores.begin() + i + 1, box_scores.end(), [](const
//         Detection &a, const Detection &b)
//                   { return a.class_prob > b.class_prob; });
//     }

//     for (auto it = picked.begin(); it != picked.end(); ++it)
//     {
//         selected_boxes.push_back(box_scores[*it]);
//     }

//     return selected_boxes;
// }

#endif // V2H