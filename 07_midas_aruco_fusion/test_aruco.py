import cv2
import sys
import os

sys.path.append(os.path.abspath("../06_aruco_marker"))
from aruco_detector import ArucoDetector
from core.moildev_applicator import MoildevApplicator

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = cap.read()
if ret:
    cv2.imwrite("/home/sushiroll/.gemini/antigravity/brain/3135cbf4-ac92-4b8f-89d3-46e8233431ab/artifacts/raw_frame.jpg", frame)
    
    # Test on raw
    detector = ArucoDetector(marker_size_cm=5.0, dictionary_name="DICT_4X4_50")
    results_raw = detector.detect(frame)
    annotated_raw = detector.annotate_frame(frame, results_raw)
    cv2.imwrite("/home/sushiroll/.gemini/antigravity/brain/3135cbf4-ac92-4b8f-89d3-46e8233431ab/artifacts/annotated_raw.jpg", annotated_raw)
    print(f"Detected {len(results_raw)} markers in raw frame")

    # Test on undistorted (zoom 1.4)
    h, w = frame.shape[:2]
    moil = MoildevApplicator(
        json_path="camera_parameters.json",
        camera_name="lrcp_imx586_240_17",
        zoom=1.4,
        frame_width=w,
        frame_height=h
    )
    undistorted = moil.undistort(frame)
    cv2.imwrite("/home/sushiroll/.gemini/antigravity/brain/3135cbf4-ac92-4b8f-89d3-46e8233431ab/artifacts/undistorted_frame.jpg", undistorted)
    
    results_undistorted = detector.detect(undistorted)
    annotated_undistorted = detector.annotate_frame(undistorted, results_undistorted)
    cv2.imwrite("/home/sushiroll/.gemini/antigravity/brain/3135cbf4-ac92-4b8f-89d3-46e8233431ab/artifacts/annotated_undistorted.jpg", annotated_undistorted)
    print(f"Detected {len(results_undistorted)} markers in undistorted frame")
else:
    print("Could not capture frame")

cap.release()
