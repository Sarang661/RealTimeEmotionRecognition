import numpy as np
import cv2
from machinelearning import pipeline_model

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if ret == False:
        break
    image,res = pipeline_model(frame)
    print(res)
    cv2.imshow('frame',frame)
    cv2.imshow('face recognition',image)
    if cv2.waitKey(1) == 27:
        break
        
cap.release()
cv2.destroyAllWindows()