"""
tests/test_camera_exposure_diag.py
===================================
Diagnostik hardware exposure kamera — jalankan TANPA GUI untuk melihat
apakah kamera benar-benar merespons perubahan CAP_PROP_EXPOSURE.
"""
import cv2
import time
import sys

CAM_IDX = 0
CAP_W, CAP_H = 2592, 1944

def mean_brightness(cap, n_flush=4):
    """Baca brightness rata-rata frame setelah flush buffer."""
    for _ in range(n_flush):
        cap.grab()
    ret, f = cap.read()
    if not ret or f is None:
        return -1.0
    return float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean())

def main():
    cap = cv2.VideoCapture(CAM_IDX)
    if not cap.isOpened():
        print(f"[ERROR] Tidak bisa membuka kamera {CAM_IDX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

    # Warmup
    time.sleep(1.5)
    for _ in range(10):
        cap.grab()

    auto_brightness = mean_brightness(cap)
    print(f"\n{'='*55}")
    print(f"  Diagnostik Exposure Kamera (V4L2)")
    print(f"{'='*55}")
    print(f"  Mode awal (Auto): brightness = {auto_brightness:.1f}")

    # Baca nilai exposure saat auto
    auto_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
    print(f"  CAP_PROP_EXPOSURE saat Auto    = {auto_exp}")
    print(f"  CAP_PROP_AUTO_EXPOSURE saat ini= {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
    print()

    # Kunci ke Manual Mode
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    time.sleep(0.3)
    print(f"  Setelah set AUTO_EXPOSURE=1: auto_exp prop = {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
    print()

    # Uji berbagai nilai exposure dan baca brightness aktual
    test_values = [500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000]
    print(f"  {'Exposure (raw)':>15} | {'Display (÷1000)':>14} | {'Brightness':>10} | {'Actual Exp Read':>15}")
    print(f"  {'-'*60}")

    results = []
    for val in test_values:
        cap.set(cv2.CAP_PROP_EXPOSURE, val)
        time.sleep(0.5)  # tunggu sensor settle
        actual_read = cap.get(cv2.CAP_PROP_EXPOSURE)
        brightness  = mean_brightness(cap, n_flush=5)
        results.append((val, actual_read, brightness))
        accepted = "✅" if abs(actual_read - val) < 200 else "❌ IGNORED"
        print(f"  {val:>15} | {val/1000:>14.1f} | {brightness:>10.1f} | {actual_read:>15.0f}  {accepted}")

    print()
    # Analisis apakah kamera merespons
    brightnesses = [r[2] for r in results]
    max_b = max(brightnesses)
    min_b = min(brightnesses)
    print(f"  Rentang brightness: {min_b:.1f} – {max_b:.1f}")
    print(f"  Delta total       : {max_b - min_b:.1f}")
    if max_b - min_b < 10:
        print(f"\n  ⚠️  KESIMPULAN: Kamera TIDAK merespons CAP_PROP_EXPOSURE!")
        print(f"     Driver V4L2 kemungkinan menggunakan skala berbeda (mis. -13 sampai 0)")
        print(f"     atau kamera ini terkunci ke driver auto yang tidak bisa dioverride.")
    elif max_b - min_b < 30:
        print(f"\n  ⚠️  KESIMPULAN: Kamera merespons LEMAH. Range efektif terbatas.")
    else:
        print(f"\n  ✅ KESIMPULAN: Kamera merespons dengan baik terhadap CAP_PROP_EXPOSURE.")

    # Coba skala V4L2 standar (-13 s/d 0)
    print(f"\n  --- Uji skala V4L2 negatif (beberapa driver pakai -13..0) ---")
    print(f"  {'Exposure':>10} | {'Brightness':>10} | {'Actual Read':>12}")
    print(f"  {'-'*38}")
    for val in [-13, -10, -7, -5, -3, -1, 0]:
        cap.set(cv2.CAP_PROP_EXPOSURE, val)
        time.sleep(0.5)
        actual_read = cap.get(cv2.CAP_PROP_EXPOSURE)
        brightness = mean_brightness(cap, n_flush=5)
        print(f"  {val:>10} | {brightness:>10.1f} | {actual_read:>12.0f}")

    cap.release()
    print(f"\n{'='*55}\n")

if __name__ == "__main__":
    main()
