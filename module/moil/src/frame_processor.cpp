/**
 * @file frame_processor.cpp
 * @brief Implementasi logika pemrosesan frame dengan akselerasi OpenCV CUDA.
 */

#include "utils/frame_processor.hpp"
#include "constants/constants.hpp"
#include "cores/mediator.hpp"
#include "helpers/alert_manager.hpp"
#include "helpers/logger.hpp"
#include "helpers/thread_pool.hpp"
#include "helpers/zone_policy.hpp"
#include "models/frame_model.hpp"
#include "utils/moil_utils.hpp"
#include "utils/moildev_applicator.hpp"
#include "utils/resolution_processor.hpp"
#include "utils/video_processor.hpp"

#include <future>
#include <gtk/gtk.h>
#include <opencv2/opencv.hpp>
#ifdef USE_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp> // Header wajib untuk remap GPU
#endif
#include <tuple>

namespace
{
    static constexpr const char *TAG = "FRAME PROCESSOR";

    // Mutex dan Cache untuk data GPU agar tidak upload berulang kali
    static std::mutex gpuMapMutex;
    using MapKey = std::tuple<int, float, float, float, int, int>;
    static std::map<MapKey, std::pair<cv::cuda::GpuMat, cv::cuda::GpuMat>> gpuMapCache;

    static std::atomic<int64_t> lastDisplayedSequence{0};
}

// Inisialisasi variabel statis
cv::Mat FrameProcessor::originalFrame;
cv::Mat FrameProcessor::originalFrameCam1;
cv::Mat FrameProcessor::originalFrameCam2;
cv::Mat FrameProcessor::scaledFrame;
cv::Mat FrameProcessor::playbackFrame;
cv::VideoCapture FrameProcessor::cap;
std::mutex FrameProcessor::frameMutex;

struct IdleData
{
    cv::Mat *frame;
    GtkImage *widget;
    int64_t sequenceId;
};

/**
 * @brief Callback GTK untuk memperbarui UI dari thread utama.
 */
static gboolean idleUpdateImage(gpointer raw)
{
    auto *d = static_cast<IdleData *>(raw);

    // CEK URUTAN: Jika frame yang datang lebih tua, buang!
    if (d->sequenceId < lastDisplayedSequence.load())
    {
        delete d->frame;
        delete d;
        return G_SOURCE_REMOVE;
    }

    // Update nomor urut terakhir yang ditampilkan
    lastDisplayedSequence.store(d->sequenceId);

    cv::Mat *mat = d->frame;

    // Konversi Mat ke GdkPixbuf (Zero-copy data pointer)
    GdkPixbuf *pix = gdk_pixbuf_new_from_data(
        mat->data,
        GDK_COLORSPACE_RGB, FALSE, 8, mat->cols, mat->rows,
        static_cast<int>(mat->step),
        [](guchar *pixels, gpointer user_data)
        {
            delete static_cast<cv::Mat *>(user_data);
        },
        mat);

    if (pix)
    {
        gtk_image_set_from_pixbuf(GTK_IMAGE(d->widget), pix);
        g_object_unref(pix);
    }

    delete d;
    return G_SOURCE_REMOVE;
}

struct BorderStyleData
{
    GtkWidget *widget;
    bool add_border;
};

static gboolean idleUpdateBorderStyle(gpointer raw)
{
    auto *data = static_cast<BorderStyleData *>(raw);
    GtkStyleContext *context = gtk_widget_get_style_context(data->widget);

    if (data->add_border)
        gtk_style_context_add_class(context, "red-border");
    else
        gtk_style_context_remove_class(context, "red-border");

    delete data;
    return G_SOURCE_REMOVE;
}

void FrameProcessor::processMultipleTasksAsync(const std::vector<FunctionProcess> &fps)
{
    ThreadPool *pool = Mediator::getInstance().threadPool.get();
    if (!pool)
        return;

    // Proteksi queue agar tidak terjadi memory leak saat lag
    if (pool->getQueueSize() > 8)
    {
        return;
    }

    for (auto const &fp : fps)
    {
        pool->enqueue([fp]()
                      { FrameProcessor::processFramesAsynchronously(fp); });
    }
}

void FrameProcessor::processFramesAsynchronously(const FunctionProcess &fp)
{
    if (fp.frame.empty())
        return;

    cv::Mat finalFrame;
    const auto &centers = fp.detection.centers;

    // --- 1. VIEW ORIGINAL (FISHEYE) ---
    if (fp.viewType == ViewType::ORIGINAL)
    {
        cv::Mat workMat = fp.frame;

        if (fp.functionRecord.id != 0 && !fp.functionRecord.mapX.empty() &&
            VideoProcessor::currentView >= VideoProcessor::VideoType::CONFIGURATION_ANYPOINT_1)
        {
            workMat = fp.frame.clone();
            MoilUtils::drawPolygon(workMat, fp.functionRecord.mapX, fp.functionRecord.mapY);
        }
        else if (!centers.empty() && VideoProcessor::showCircleIndicators)
        {
            for (const auto &center : centers)
                cv::circle(workMat, center, 20, cv::Scalar(0, 255, 255), -1);
        }
        finalFrame = FrameProcessor::resizeFrame(workMat, cv::Size(fp.frameRecord.width, fp.frameRecord.height));
    }
    // --- 2. VIEW PANORAMA / ANYPOINT ---
    else if (fp.viewType == ViewType::PANORAMA || fp.viewType == ViewType::ANYPOINT)
    {
        // 1. LANGKAH AWAL: Remap
        cv::Mat tempFrame = remapFrame(fp.frame, fp.frameRecord, fp.functionRecord);

        // 2. LANGKAH KEDUA: Jalankan Flip TERLEBIH DAHULU
        int vFlip = 0;
        int hFlip = 0;
        if (fp.functionRecord.id != 0)
        {
            vFlip = fp.functionRecord.flipState;
            hFlip = fp.functionRecord.flipHState;
            if (vFlip == 1 || hFlip == 1)
            {
                int code = (vFlip == 1 && hFlip == 1) ? -1 : (vFlip == 1 ? 0 : 1);
                cv::flip(tempFrame, tempFrame, code);
            }
        }

        // 3. LANGKAH KETIGA: Gambar Deteksi (SETELAH FLIP)
        if (VideoProcessor::currentAiState)
        {
            if (fp.viewType == ViewType::PANORAMA)
            {
                // Gunakan konstanta AI_INPUT agar presisi
                float scaleX = (float)tempFrame.cols / 640.0f;
                float scaleY = (float)tempFrame.rows / 240.0f;

                for (size_t i = 0; i < fp.detection.boxes.size(); ++i)
                {
                    cv::Rect r = fp.detection.boxes[i];

                    // --- LOGIKA KOMPENSASI KOORDINAT KOTAK ---
                    // Karena YOLO melihat gambar yang SUDAH di-flip di runAiInferenceAsync,
                    // dan sekarang kita menggambar di tempFrame yang JUGA SUDAH di-flip,
                    // koordinat r.x dan r.y dari YOLO seharusnya sudah langsung sinkron.

                    cv::Rect scaledRect(
                        static_cast<int>(r.x * scaleX),
                        static_cast<int>(r.y * scaleY),
                        static_cast<int>(r.width * scaleX),
                        static_cast<int>(r.height * scaleY));
                    cv::rectangle(tempFrame, scaledRect, fp.detection.colors[i], 3);
                }
            }
            else
            {
                // Untuk Anypoint, karena kita pakai titik pusat (centers) yang sudah
                // dipetakan balik ke Fisheye, kita biarkan logic aslinya.
                for (const auto &c : centers)
                    cv::circle(tempFrame, c, 20, cv::Scalar(0, 255, 255), -1);
            }
        }
        finalFrame = std::move(tempFrame);

        // Alert Border Logic
        if (fp.borderBox != nullptr)
        {
            const ZonePolicy &pol = fp.resolvedPolicy;
            bool insideZone = (pol.zone == Zone::FRONT) ? VRUDetector::anyInFront(centers, fp.frame.size()) : (pol.zone == Zone::RIGHT) ? VRUDetector::anyInRight(centers, fp.frame.size())
                                                                                                                                        : false;

            bool allowed = (pol.zone == Zone::FRONT) ? AlertManager::I().lowAllowedFront() : (pol.zone == Zone::RIGHT) ? AlertManager::I().lowAllowedRight()
                                                                                                                       : false;

            auto *borderData = new BorderStyleData{GTK_WIDGET(fp.borderBox), (pol.allowLowBorder && insideZone && allowed)};
            g_idle_add(idleUpdateBorderStyle, borderData);
        }
    }
    else
    {
        finalFrame = fp.frame;
    }

    // UI Handover
    cv::Mat *finalFramePtr = new cv::Mat(std::move(finalFrame));
    auto *imageData = new IdleData{finalFramePtr, fp.widget, fp.sequenceId};
    g_idle_add(idleUpdateImage, imageData);
}

void FrameProcessor::clearGpuMapCache()
{
    std::lock_guard<std::mutex> lock(gpuMapMutex);
    gpuMapCache.clear();
    Logger::info(TAG, "CUDA cache cleared.");
}

cv::Mat FrameProcessor::remapFrame(const cv::Mat &frame, const FrameRecord &frameRecord, const FunctionRecord &functionRecord)
{
    if (functionRecord.mapX.empty() || functionRecord.mapY.empty())
        return frame;

    cv::Mat remappedFrame;
#ifdef USE_CUDA
    // JALUR CUDA (GPU)
    if (cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        try
        {
            cv::cuda::GpuMat d_src, d_dst, d_mapX, d_mapY;
            d_src.upload(frame);

            // Manajemen Cache VRAM
            MapKey key = {functionRecord.id, functionRecord.alpha, functionRecord.beta,
                          functionRecord.zoom, functionRecord.mapX.cols, functionRecord.mapX.rows};

            {
                std::lock_guard<std::mutex> lock(gpuMapMutex);
                auto it = gpuMapCache.find(key);
                if (it != gpuMapCache.end())
                {
                    d_mapX = it->second.first;
                    d_mapY = it->second.second;
                }
                else
                {
                    d_mapX.upload(functionRecord.mapX);
                    d_mapY.upload(functionRecord.mapY);
                    gpuMapCache[key] = {d_mapX, d_mapY};
                }
            }

            cv::cuda::remap(d_src, d_dst, d_mapX, d_mapY, cv::INTER_LINEAR);
            d_dst.download(remappedFrame);
        }
        catch (const cv::Exception &e)
        {
            Logger::error(TAG, "CUDA Error: %s. Falling back to CPU.", e.what());
            cv::remap(frame, remappedFrame, functionRecord.mapX, functionRecord.mapY, cv::INTER_LINEAR);
        }
    }
    else // JALUR CPU
    {
        cv::remap(frame, remappedFrame, functionRecord.mapX, functionRecord.mapY, cv::INTER_LINEAR);
    }
#else
    // 🔵 JALUR CPU - Di-compile jika OpenCV CUDA tidak ditemukan di sistem
    cv::remap(frame, remappedFrame, functionRecord.mapX, functionRecord.mapY, cv::INTER_LINEAR);
#endif

    // Panorama Stretch
    if (frameRecord.name.find("Panorama") != std::string::npos)
    {
        cv::resize(remappedFrame, remappedFrame, cv::Size(remappedFrame.cols * 2, remappedFrame.rows), 0, 0, cv::INTER_LINEAR);
    }

    return fit_and_stretch_single_pass(remappedFrame, frameRecord.width, frameRecord.height, false);
}

cv::Mat FrameProcessor::resizeFrame(const cv::Mat &frame, const cv::Size &Dimension)
{
    cv::Mat resizedFrame;
    float scale = std::min((float)Dimension.width / frame.cols, (float)Dimension.height / frame.rows);

    if (scale != 1.0)
        cv::resize(frame, resizedFrame, cv::Size(), scale, scale, cv::INTER_AREA);
    else
        resizedFrame = frame.clone();

    return resizedFrame;
}

cv::Mat FrameProcessor::fit_and_stretch_single_pass(const cv::Mat &source, int target_w, int target_h, bool stretchX2)
{
    if (source.empty())
        return cv::Mat();

    int logical_w = stretchX2 ? source.cols * 2 : source.cols;
    int logical_h = source.rows;

    double scale = std::max((double)target_w / logical_w, (double)target_h / logical_h);
    int physical_new_w = static_cast<int>(logical_w * scale);
    int physical_new_h = static_cast<int>(logical_h * scale);

    cv::Mat resized;
    cv::resize(source, resized, cv::Size(physical_new_w, physical_new_h), 0, 0, cv::INTER_LINEAR);

    int crop_x = (physical_new_w - target_w) / 2;
    int crop_y = (physical_new_h - target_h) / 2;

    return resized(cv::Rect(crop_x, crop_y, target_w, target_h)).clone();
}