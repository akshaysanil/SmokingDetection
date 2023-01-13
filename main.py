import torch
import numpy as np
import cv2


model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt',  force_reload=True)
model.conf = 0.6

cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi',fourcc,15,(640,480))


while cap.isOpened():
    ret, frame = cap.read()
    result = model(frame)


    cv2.imshow('window', np.squeeze(result.render()))
    out.write(np.squeeze(result.render()))
    if cv2.waitKey(10) == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()

# img = model('smoking5.jpg')
# img.save()
# print(img)