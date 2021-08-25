import numpy as np
import cv2
frame = cv2.imread()
def mask(dets,frame):
    w_,h_, c_ = frame.shape
    mask_blank = np.ones((w_,h_))
    for d in dets:
        x1 = max(0, int(d[0]))
        y1 = max(0, int(d[1]))
        x2 = min(w_-1, int(d[2]))
        y2 = min(h_-1, int(d[3]))
        mask_blank[y1:y2,x1:x2] = mask_blank[y1:y2,x1:x2] + 1