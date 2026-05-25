#include <detections/ai.h>
#include <unistd.h>

#ifdef V2H
uint64_t get_drpai_start_addr(int drpai_fd) {
    int ret = 0;
    drpai_data_t drpai_data;

    errno = 0;

    /* Get DRP-AI Memory Area Address via DRP-AI Driver */
    ret = ioctl(drpai_fd, DRPAI_GET_DRPAI_AREA, &drpai_data);
    if (-1 == ret) {
        LOGR_ERROR("Failed to get DRP-AI Memory Area : errno=" + std::to_string(errno));
        return 0;
    }

    return drpai_data.address;
}

uint64_t init_drpai(int drpai_fd) {
    int ret = 0;
    drpai_data_t drpai_data;

    /*Get DRP-AI memory start address*/
    ret = ioctl(drpai_fd, DRPAI_GET_DRPAI_AREA, &drpai_data);
    if (ret != 0) {
        LOGR_ERROR("Failed to get DRP-AI Memory Area");
        return 0;
    }

    return drpai_data.address;
}
#endif

void AI::ReloadPre(const std::string& dir) {
#if defined(V2H) && (DRP_AI_TVM_RUNTIME == 1)
    if (current_pre_dir == dir) return;
    
    LOGR_INFO("Switching DRP-AI Pre-processing to: " + dir);
    shared_preruntime->Load(dir);
    current_pre_dir = dir;
#endif
}

AI::AI() {
    // Initialize Hardware Device
#ifdef V2H
    drpai_fd = open("/dev/drpai0", O_RDWR);
    if (drpai_fd < 0) {
        LOGR_ERROR("Failed to open /dev/drpai0");
    }

    drpaimem_addr_start = init_drpai(drpai_fd);

    if (drpaimem_addr_start == 0) {
        LOGR_ERROR("Failed initialize drp driver");
    }

#if (1) == DRP_AI_TVM_RUNTIME
    // ── Initialize Shared PreRuntime ──
    shared_preruntime = std::make_unique<PreRuntime>();

    // ── Initialize Cup Detector (YOLOv8) ──
    // Load YOLOv8 Pre-processing (initial)
    ReloadPre(c::yolov8::pre_dir);
    
    cup_runtime = std::make_unique<MeraDrpRuntimeWrapper>();
    cup_runtime_status = cup_runtime->LoadModel(
        c::yolov8::model_dir,
        drpaimem_addr_start + c::yolov8::DRPAI_MEM_OFFSET);
    
    if (!cup_runtime_status) {
        LOGR_ERROR("failed to load model yolov8");
    }

    cup_detector = std::make_unique<CupDetectorV2H>();
    cup_detector->runtime    = cup_runtime.get();
    cup_detector->preruntime = shared_preruntime.get();

    // Prepare MiDaS Estimator object, but don't load hardware model yet
    midas_estimator = std::make_unique<MidasEstimatorV2H>();
    midas_estimator->preruntime = shared_preruntime.get();
    is_midas_loaded = false;

#else
    cup_detector = std::make_unique<CupDetector>();
#endif

#else
    // Default x86 implementations
    cup_detector    = std::make_unique<CupDetector>();
    midas_estimator = std::make_unique<MidasEstimator>();
#endif
}

bool AI::LoadMidas() {
#if defined(V2H) && (DRP_AI_TVM_RUNTIME == 1)
    if (is_midas_loaded) return true;

    LOGR_INFO("Lazy-loading MiDaS model into DRP-AI...");
    
    /* We must reload Pre-processing for MiDaS during initial load */
    ReloadPre(c::midas::pre_dir);

    midas_runtime = std::make_unique<MeraDrpRuntimeWrapper>();
    midas_runtime_status = midas_runtime->LoadModel(
        c::midas::model_dir,
        drpaimem_addr_start + MIDAS_DRPAI_MEM_OFFSET);

    if (!midas_runtime_status) {
        LOGR_ERROR("failed to load MiDaS model");
        return false;
    }

    midas_estimator->runtime = midas_runtime.get();
    is_midas_loaded = true;
    LOGR_INFO("MiDaS model loaded successfully.");
    return true;
#else
    return true;
#endif
}

AI::~AI() {
#ifdef V2H
    if (drpai_fd >= 0) {
        close(drpai_fd);
        drpai_fd = -1;
    }
#endif
}