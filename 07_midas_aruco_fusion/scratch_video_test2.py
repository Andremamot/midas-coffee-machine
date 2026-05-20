import cv2
import numpy as np
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test_noise.avi', fourcc, 20.0, (2592, 1944))
for i in range(10):
    frame = np.random.randint(0, 256, (1944, 2592, 3), dtype=np.uint8)
    out.write(frame)
out.release()
print("Success with noise")
