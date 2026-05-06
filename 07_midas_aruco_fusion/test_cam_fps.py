import cv2
import time

def test_cam(fourcc_str=None):
    cap = cv2.VideoCapture(0)
    if fourcc_str:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_str))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
    
    # Warmup
    for _ in range(5): cap.read()
    
    start = time.time()
    frames = 0
    while time.time() - start < 3.0:
        ret, frame = cap.read()
        if ret: frames += 1
    
    cap.release()
    print(f"Format: {fourcc_str if fourcc_str else 'Default'} | FPS: {frames / 3.0:.2f}")

test_cam()
test_cam('MJPG')
