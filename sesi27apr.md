Edited live_pipeline.cpp

Berikut adalah rangkuman teknis dari sesi refaktorisasi MiDaS ke C++ untuk target Renesas RZ/V2H. Dokumen ini dirancang agar Sesi Berikutnya dapat langsung memahami status terakhir dan tantangan yang tersisa.

# Rangkuman Refaktorisasi MiDaS & Fusion (RZ/V2H)

## 1. Objektif Sesi
Migrasi modul `07_midas_aruco_fusion` dari Python ke C++ dengan integrasi hardware acceleration DRP-AI (Renesas RZ/V2H). Mengganti ketergantungan OpenCV DNN dengan DRP-AI TVM Runtime untuk MiDaS dan YOLOv8.

## 2. Arsitektur yang Diimplementasikan
*   **AI Singleton**: Satu titik kontrol (`AI::get_instance()`) yang mengelola dua model secara bersamaan: `CupDetectorV2H` (YOLOv8) dan `MidasEstimatorV2H` (MiDaS).
*   **Model Isolation**: Memisahkan memory area DRP-AI untuk kedua model menggunakan offset fisik:
    *   Cup Detector: `0x0`
    *   MiDaS: `0x8000000` (128 MiB offset, telah ditingkatkan untuk keamanan).
*   **DRPQueue**: Implementasi sistem antrean (serialization) untuk memastikan hanya satu tugas DRP-AI (Pre-processor atau Inference) yang berjalan pada satu waktu di satu hardware unit.

## 3. Perbaikan Build & Linker
*   Penambahan library sistem `mmngr` dan `mmngrbuf` pada `CMakeLists.txt` untuk mendukung flush buffer DMA.
*   Perbaikan path include ArUco dan MiDaS pada sub-modul fusion agar tidak bergantung pada relative path yang dalam.
*   Penerapan global flag `V2H` dan `DRP_AI_TVM_RUNTIME` pada seluruh proyek.

## 4. Troubleshooting Hardware (Status Terkini)
Meskipun antrean sudah diterapkan, sistem masih mengalami crash `DRPAI_GET_STATUS failed` saat kamera mulai mendeteksi ArUco (titik di mana inferensi YOLO pertama kali dipicu).

### Solusi Stabilitas yang Telah Dicoba:
1.  **Memory Alignment**: Memastikan `MIDAS_DRPAI_MEM_OFFSET` sejajar dengan 16 MiB (syarat mutlak V2H).
2.  **Frequency Tuning**: Menurunkan frekuensi DRP-AI dari 1 GHz (`FREQ=2`) ke 630 MHz (`FREQ=3`) untuk stabilitas thermal.
3.  **Initialization Delay**: Menambahkan `usleep(200000)` di antara proses `LoadModel` MiDaS dan YOLO untuk mencegah kondisi balapan pada driver.
4.  **Implicit Assignment**: Menggunakan `ioctl` untuk mendapatkan delegasi area memori DRP-AI sebelum loading model.

## 5. Hipotesis & Kendala yang Tersisa (Untuk Sesi Depan)
*   **Resource Hijacking**: Ada kemungkinan penggunaan dua instansi `MeraDrpRuntimeWrapper` yang membuka `/dev/drpai0` secara terpisah tetap menyebabkan konflik status driver di level kernel, meskipun sudah diantre di level aplikasi.
*   **Context Collision**: Perlu dipastikan apakah MiDaS memerlukan area memori yang lebih besar atau apakah file `.so` model MiDaS memiliki konfigurasi pre-processing yang bertabrakan dengan YOLO di level hardware pre-processor.
*   **Isolation Test**: Langkah selanjutnya yang disarankan adalah mencoba menjalankan MiDaS secara mandiri (tanpa YOLO) untuk memverifikasi integritas runtime MiDaS pada V2H.

---
**File Penting:**
*   `module/detections/src/ai.cpp`: Logika inisialisasi ganda.
*   `module/detections/include/detections/drp_queue.h`: Mekanisme antrean hardware.
*   `module/detections/include/constants/define_midas.h`: Konfigurasi memory offset.