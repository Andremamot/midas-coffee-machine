import cv2
import time
import pytest

def test_camera_fps_is_acceptable():
    """
    Test that the camera can capture frames at an acceptable FPS (>10.0) 
    at native 2592x1944 resolution.
    """
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # TDD Fix applied
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    
    # Warmup
    for _ in range(5): cap.read()
    
    start = time.time()
    frames = 0
    while time.time() - start < 3.0:
        ret, frame = cap.read()
        if ret: frames += 1
    
    fps = frames / 3.0
    cap.release()
    
    assert fps > 10.0, f"FPS is too low: {fps:.2f}. Expected > 10.0 FPS. Check USB bandwidth or codec (MJPG vs YUYV)."
