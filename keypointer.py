import torch
import torchvision
from torchvision import models
import torchvision.transforms as T
import cv2
import numpy as np


trf = T.Compose([
    T.ToTensor()
])

THRESHOLD = 0.95 #Accuracy
neck = np.array([])
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()

cap = cv2.VideoCapture(0) #this will be usually your internal Camera
#cap = cv2.VideoCapture(1) #Your other Camera

while cap.isOpened():

    success, img = cap.read()
    input_frame = trf(img)
    out = model([input_frame])[0]

    for score, keypoints in zip(out['scores'], out['keypoints']):
        score = score.detach().numpy()
        if score < THRESHOLD: 
            continue

        box = box.detach().numpy()
        keypoints = keypoints.detach().numpy()[:, :2]

        neck = 0.5 * (keypoints[5]+keypoints[6]) #added neck keypoint
        keypoints = np.append(keypoints, neck)



    #body   
        # left shoulder - neck
        cv2.line(img,  tuple(keypoints[10:12]), tuple(keypoints[34:36]), (0,255,0), 2)
        # right shoulder - neck
        cv2.line(img,  tuple(keypoints[12:14]), tuple(keypoints[34:36]), (0,255,0), 2)
        #neck - nose
        cv2.line(img,  tuple(keypoints[0:2]), tuple(keypoints[34:36]), (0,255,0), 2)
        #left body
        cv2.line(img,  tuple(keypoints[10:12]), tuple(keypoints[22:24]), (0,255,0), 2)
        #right body
        cv2.line(img,  tuple(keypoints[12:14]), tuple(keypoints[24:26]), (0,255,0), 2)
        #hip
        cv2.line(img,  tuple(keypoints[22:24]), tuple(keypoints[24:26]), (0,255,0), 2)

    #arm
        #left arm_up
        cv2.line(img,  tuple(keypoints[10:12]), tuple(keypoints[14:16]), (0,255,0), 2)
        #left arm_down
        cv2.line(img,  tuple(keypoints[14:16]), tuple(keypoints[18:20]), (0,255,0), 2)
        #right arm_up
        cv2.line(img,  tuple(keypoints[12:14]), tuple(keypoints[16:18]), (0,255,0), 2)
        #right arm_down
        cv2.line(img,  tuple(keypoints[16:18]), tuple(keypoints[20:22]), (0,255,0), 2)
        
    
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
        #left eye - nose
        cv2.line(img,  tuple(keypoints[0:2]), tuple(keypoints[2:4]), (0,255,0), 2)
        #right eye - nose
        cv2.line(img,  tuple(keypoints[0:2]), tuple(keypoints[4:6]), (0,255,0), 2)
        #left ear - leye
        cv2.line(img,  tuple(keypoints[2:4]), tuple(keypoints[6:8]), (0,255,0), 2)
        #right eye - reye
        cv2.line(img,  tuple(keypoints[4:6]), tuple(keypoints[8:10]), (0,255,0), 2)

    cap.release()
    
    
    
'''
Index

keypointindex   'bodyname' , tupleindex-x, tupleindex-y

0   'nose',0,1
1   'left_eye',2,3
2   'right_eye',4,5
3   'left_ear',6,7
4   'right_ear',8,9
5    'left_shoulder',10,11
6     'right_shoulder',12,13
7     'left_elbow',14,15
8     'right_elbow',16,17
9     'left_wrist',18,19
10    'right_wrist',20,21
11    'left_hip',22,23
12    'right_hip',24,25
13    'left_knee',26,27
14    'right_knee',28,29
15    'left_ankle',30,31
16    'right_ankle',32,33
17    'neck',34,35
