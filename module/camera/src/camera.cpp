#include <camera/camera.h>
#include <cstdlib>

/* =========================
 * Constructor / Destructor
 * ========================= */
Camera::Camera(std::variant<int, std::string> source, bool autostart, int manual_exposure)
    : camera_source(source), running(false), exposure_val_(manual_exposure) {
  if (autostart) {
    start_camera();
  }
}

Camera::~Camera() { stop_camera(); }

/* =========================
 * Control
 * ========================= */
void Camera::start_camera() {
  // Do nothing if already running
  if (running.load())
    return;

  running = true;
  camera_thread = std::thread(&Camera::camera_loop, this);
}

void Camera::stop_camera() {
  if (!running.load())
    return;

  running = false;

  if (camera_thread.joinable()) {
    camera_thread.join();
  }

  if (cap.isOpened()) {
    cap.release();
  }

  // Clear last frame
  {
    std::lock_guard<std::mutex> lock(frame_mutex);
    frame.release();
    frame = cv::Mat();
  }
}

void Camera::set_smart_exposure(float val) {
  if (val < 1.0f) val = 1.0f;
  if (val > 10.0f) val = 10.0f;
  
  // Exposure time (Raw): 1000 -> 10000
  int raw_exp = static_cast<int>(std::round(val * 1000.0f));
  
  // Gain (0 - 255): Scale linearly from val 1.0 to 10.0
  int raw_gain = static_cast<int>(std::round((val - 1.0f) / 9.0f * 255.0f));
  
  // Brightness EV (-64 to +64): Scale linearly
  int raw_bri = static_cast<int>(std::round((val - 1.0f) / 9.0f * 128.0f - 64.0f));

  exposure_val_.store(raw_exp); // Simpan exposure raw jika dibutuhkan

  int idx = 0;
  if (std::holds_alternative<int>(camera_source)) {
    idx = std::get<int>(camera_source);
  }
  
  std::string cmd = "v4l2-ctl -d /dev/video" + std::to_string(idx) +
                    " --set-ctrl=exposure_auto=1" +
                    " --set-ctrl=exposure_absolute=" + std::to_string(raw_exp) +
                    " --set-ctrl=gain=" + std::to_string(raw_gain) +
                    " --set-ctrl=brightness=" + std::to_string(raw_bri) +
                    " >/dev/null 2>&1";
  std::system(cmd.c_str());
}

/* =========================
 * Frame access
 * ========================= */
cv::Mat Camera::get_frame() {
  std::lock_guard<std::mutex> lock(frame_mutex);

  if (frame.empty()) {
    return cv::Mat();
  }

  cv::Mat resized_frame;

#ifdef V2H
#if INPUT_CAM_TYPE == 1
  cv::resize(frame, resized_frame, cv::Size(MIPI_WIDTH, MIPI_HEIGHT));
#else
  cv::resize(frame, resized_frame, cv::Size(USB_WIDTH, USB_HEIGHT));
#endif // INPUT_CAM_TYPE
#else
  resized_frame = frame;
#endif // V2H
  return resized_frame;
}

/* =========================
 * Internal capture loop
 * ========================= */
void Camera::camera_loop() {
  while (running.load()) {
    // Build GStreamer pipeline for the device
    std::string index = std::to_string(std::get<int>(camera_source));
    std::string device = "/dev/video" + index;
    // std::string pipeline = "v4l2src device=" + device +
    //                        " io-mode=0 do-timestamp=true "
    //                        "!
    //                        video/x-raw,format=YUY2,width=320,height=240,framerate=30/1
    //                        "
    //                        "! queue max-size-buffers=2 leaky=downstream "
    //                        "! videoconvert ! video/x-raw,format=BGR "
    //                        "! queue max-size-buffers=2 leaky=downstream "
    //                        "! appsink max-buffers=1 drop=true sync=false
    //                        wait-on-eos=false";
    std::string pipeline = "v4l2src device=" + device +
                           " io-mode=2 do-timestamp=true "
                           "! image/jpeg,width=2592,height=1944,framerate=20/1 "
                           "! jpegdec "
                           "! videoconvert "
                           "! video/x-raw,format=BGR "
                           "! appsink max-buffers=1 drop=true sync=false";

    bool opened = cap.open(pipeline, cv::CAP_GSTREAMER);

    if (!opened || !cap.isOpened()) {
      LOGR_WARN("Failed to open camera. Retrying in 1 second...");
      std::this_thread::sleep_for(std::chrono::seconds(1));
      continue;
    }

    LOGR_INFO("Camera opened successfully.");

    // Skip initial frames for stabilization
    cv::Mat skip_frame;
    for (int i = 0; i < 20 && running.load(); ++i) {
      cap >> skip_frame;
      std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    // Main capture loop
    cv::Mat local_frame;
    while (running.load()) {
      if (!cap.read(local_frame) || local_frame.empty()) {
        LOGR_WARN("Empty frame received, restarting capture...");
        cap.release();
        break;
      }

      // Update internal frame buffer
      {
        std::lock_guard<std::mutex> lock(frame_mutex);
        local_frame.copyTo(frame);
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    cap.release();

    {
      std::lock_guard<std::mutex> lock(frame_mutex);
      frame.release();
    }
  }

  LOGR_INFO("Camera capture loop ended.");
}