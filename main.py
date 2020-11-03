import torch
import torchvision
from torchvision import models
import torchvision.transforms as T
import cv2

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

trf = T.Compose([
    T.ToTensor()
])

THRESHOLD = 0.97
neck = np.array([])
model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()


'''
#Photo
img = cv2.imread('imgs/example1.jpg', cv2.IMREAD_COLOR)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, dsize=(690,545), interpolation=cv2.INTER_AREA)
'''



'''
#Video
while(1):
    cap = cv2.VideoCapture(0)
    if cap.isOpen():
    	print('width: {}, height : {}'.format(cap.get(3), cap.get(4))
    
    while True:
    	ret, fram = cap.read()
    
    	if ret:
    		gray = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)
    		cv2.imshow('video', gray)
    		k == cv2.waitKey(1) & 0xFF
    		if k == 27: 
    			break
    	else:
    		print('error')
'''


cap = cv2.VideoCapture('imgs/02.mp4')

while(cap.isOpened()):

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)




    input_frame = trf(img) 

    out = model([input_frame])[0] 

    for box, score, keypoints in zip(out['boxes'], out['scores'], out['keypoints']):
        score = score.detach().numpy() 

        if score < THRESHOLD: 
            continue

        box = box.detach().numpy()
        keypoints = keypoints.detach().numpy()[:, :2]

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

    cap.release()
