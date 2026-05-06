# 07 MiDaS + ArUco Fusion — Estimasi Tinggi Gelas

Modul *standalone* yang menggabungkan **ArUco Marker** (jarak absolut), **MiDaS** (kedalaman relatif), dan **YOLOv8** (deteksi multi-gelas) untuk mengestimasi tinggi gelas secara geometris. Sistem terbaru ini dilengkapi dengan penyimpanan menu profil (*Menu-Driven*), kompensasi lensa *Fisheye*, serta pelacakan *Multi-Cup* secara simultan.

## Prasyarat Lingkungan

```bash
conda env create -f environment_fusion.yaml
conda activate midas-aruco-env
```

## Alur Penggunaan (Workflow)

Sistem dirancang untuk beroperasi di pabrik dalam 3 fase utama:

### 1. Kalibrasi Lensa Fisheye (Khusus Kamera Fisheye / Via Moildev)
Jika memakai kamera fisheye berekstensi lebar seperti **LRCP + IMX586**, distorsi barrel dikoreksi secara real-time menggunakan library **Moildev** dengan database parameter `camera_parameters.json`.

```bash
# Tidak perlu kalibrasi manual! Profil sudah tersedia di camera_parameters.json.
# Cukup jalankan langsung dengan --fisheye dan nama profil yang sesuai:
python run_fusion.py --fisheye --moil-camera-name syue_7730v1_6 --calibrate 5 --true-height 7.6
```

> **Catatan**: Untuk kamera biasa (tanpa fisheye), **hapus flag `--fisheye`**. Moildev tidak aktif dan pipeline berjalan seperti biasa.

### Uji Coba Mode Anypoint (Baru!)
Anda dapat membandingkan **Mode 1 (Alpha/Beta)** dan **Mode 2 (Pitch/Yaw/Roll)** secara visual sebelum menjalankan kalibrasi menggunakan script interaktif:
```bash
python compare_anypoint_modes.py --camera 0 --camera-name syue_7730v1_6
```

### 2. Pendaftaran Profil Mangkuk (Kalibrasi Mode 5)
Latih program di mana posisi aman gelas berukuran tertentu (misal: menu gelas `7.6` cm). Program akan merekam persentase *clipping* kedalaman secara otomatis (*PolyFit*).

```bash
python run_fusion.py --camera 0 --fisheye --calibrate 5 --n-positions 3 --true-height 7.6
```
- Gerakkan moncong wadah secara bertahap `n` kali ikuti instruksi yang tertulis di layar HUD kamera.
- Database akan disimpan langsung secara aman ke `calibration_fisheye_default.json`. (Ulangi perintah di atas untuk gelas ke-2 dengan `--true-height 10.1` dsb.).

### 3. Mode Operasional (Live)
Fase *Deployment*! Mode deteksi performa tinggi (*Fast Track* & *Heavy Track*) tanpa jeda untuk melihat hingga 2 gelas sekaligus!

```bash
python run_fusion.py --camera 0 --fisheye --target-cup 7.6
```
- Program otomatis membuka algoritma gelas berbasis referensi ukuran 7.6 cm.
- `S`: Tembak Tangkapan Layar *(Screenshot)*.
- `R`: Toggle Rekam / Matikan Video.
- *Status Teks*: `Z_tray` (Jarak meja ke kamera), `Z_rim` (Prediksi jarak bibir gelas), dan Tinggi Bersih Gelas (dalam sentimeter).

---

## Daftar Argumen Lengkap (`run_fusion.py`)

| Argumen | Default | Deskripsi Instruksi |
|---|---|---|
| `--camera` | `0` | Kode *port* kamera video |
| `--marker-size` | `5.0` | Sisi ArUco di meja (sentimeter) |
| `--calibrate` | `0` | Isi angka `5` untuk membuat profil kalibrasi |
| `--true-height` | `None` | (Kalibrasi) Masukkan tinggi fisik gelas aslinya |
| `--target-cup` | `None` | (Produksi Live) Ukuran referensi *Cup* mana yang ditarik |
| `--cup-profile` | `default` | Nama profil simpanan (misal `--cup-profile mesinB`) |
| `--fisheye` | `False` | Aktifkan mode Moildev fisheye undistortion |
| `--moil-camera-name` | `syue_7730v1_6` | Nama profil di `camera_parameters.json` |
| `--moil-mode` | `2` | Mode anypoint: `1` (Alpha/Beta) atau `2` (Pitch/Yaw/Roll) |
| `--moil-pitch` | `0.0` | Anypoint pitch / Alpha (°) — geser dengan drag atas/bawah |
| `--moil-yaw` | `0.0` | Anypoint yaw / Beta (°) — geser dengan drag kiri/kanan |
| `--moil-roll` | `0.0` | Anypoint roll (°) — hanya di Mode 2 |
| `--moil-zoom` | `1.4` | Zoom factor anypoint Moildev — geser dengan roda mouse |
| `--n-positions` | `3` | Hitungan fase titik berhenti (*Z-Grid*) yang harus dipelajari |
| `--headless` | `False` | Sembunyikan *Window* saat mesin ditanam di `Kakip RZ/V2H` |

## Hasil Akhir Perjalanan (Automatic Session Report)
Setelah menekan tombol **`Q`** atau mematikannya di terminal (`Ctrl+C`), ia akan mengenerate folder baru secara otomatis di bawah direktori `results/report/YYYY-MM-DD_HH-MM-SS/`:
- Menghasilkan gambar Grafik Ketinggian Multi-Gelas (`session_chart.png`).
- Mengenerate ringkasan ketepatan simpangan AI / varians deviasi (`report.md`).
- Mencadangkan *Log* kasar di `session_data.json` persis pada masing-masing iterasi frame.
