import cv2
import numpy as np
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test.avi', fourcc, 20.0, (2592, 1944))
frame = np.zeros((1944, 2592, 3), dtype=np.uint8)
out.write(frame)
out.release()
print("Success")
