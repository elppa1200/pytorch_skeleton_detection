import torch, torchvision, cv2
from torchvision import models
import torchvision.transforms as T
import numpy as np


trf = T.Compose([
    T.ToTensor()
])

a = 30
b = -30

THRESHOLD = 0.95
neck = np.array([])

model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 940)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():

    success, img = cap.read()
    input_frame = trf(img)
    out = model([input_frame])[0]

    for box, score, keypoints in zip(out['boxes'],out['scores'], out['keypoints']): 
       
        score = score.detach().numpy()
        box = box.detach().numpy()
        keypoints = keypoints.detach().numpy()[:, :2]

        if score < THRESHOLD: 
            continue

        neck = 0.5 * (keypoints[5]+keypoints[6]) 
        keypoints = np.append(keypoints, neck)

    
        #body   
        # lsh - neck
        cv2.line(img,  tuple(keypoints[10:12]), tuple(keypoints[34:36]), (0,255,0), 2)
        # rsh - neck
        cv2.line(img,  tuple(keypoints[12:14]), tuple(keypoints[34:36]), (0,255,0), 2)
        #neck - nose
        cv2.line(img,  tuple(keypoints[0:2]), tuple(keypoints[34:36]), (0,255,0), 2)
        # left body
        cv2.line(img,  tuple(keypoints[10:12]), tuple(keypoints[22:24]), (0,255,0), 2)
        # right body
        cv2.line(img,  tuple(keypoints[12:14]), tuple(keypoints[24:26]), (0,255,0), 2)
        # hip
        cv2.line(img,  tuple(keypoints[22:24]), tuple(keypoints[24:26]), (0,255,0), 2)
    
        #arm
        # left arm_down
        cv2.line(img,  tuple(keypoints[14:16]), tuple(keypoints[18:20]), (0,255,0), 2)
        # right arm_up
        cv2.line(img,  tuple(keypoints[12:14]), tuple(keypoints[16:18]), (0,255,0), 2)
        # right arm_down
        cv2.line(img,  tuple(keypoints[16:18]), tuple(keypoints[20:22]), (0,255,0), 2)
        # left arm_up
        cv2.line(img,  tuple(keypoints[10:12]), tuple(keypoints[14:16]), (0,255,0), 2)
        
        #leg
        # left leg_up
        cv2.line(img,  tuple(keypoints[22:24]), tuple(keypoints[26:28]), (0,255,0), 2)
        # left leg_down
        cv2.line(img,  tuple(keypoints[26:28]), tuple(keypoints[30:32]), (0,255,0), 2)
        # right leg_up
        cv2.line(img,  tuple(keypoints[24:26]), tuple(keypoints[28:30]), (0,255,0), 2)
        # right leg_down
        cv2.line(img,  tuple(keypoints[28:30]), tuple(keypoints[32:34]), (0,255,0), 2)
    
        #head 
        #leye - nose
        cv2.line(img,  tuple(keypoints[0:2]), tuple(keypoints[2:4]), (0,255,0), 2)
        #reye - nose
        cv2.line(img,  tuple(keypoints[0:2]), tuple(keypoints[4:6]), (0,255,0), 2)
        #lear - leye
        cv2.line(img,  tuple(keypoints[2:4]), tuple(keypoints[6:8]), (0,255,0), 2)
        #reye - reye
        cv2.line(img,  tuple(keypoints[4:6]), tuple(keypoints[8:10]), (0,255,0), 2)

    cv2.imshow("Mask Detection", img)
    cv2.waitKey(1)
