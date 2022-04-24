# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 07:01:58 2022

@author: jagpreet
"""

import dlib
import matplotlib.pyplot
import numpy as np
import cv2
face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
img_path = "images/obama.jpg"
img = dlib.load_rgb_image(img_path)
faces = face_detector(img, 1)
landmark_tuple = []
for k, d in enumerate(faces):
   landmarks = landmark_detector(img, d)
   for n in range(0, 68):
      x = landmarks.part(n).x
      y = landmarks.part(n).y
      landmark_tuple.append((x, y))
      cv2.circle(img, (x, y), 2, (255, 255, 0), -1)

routes = []
 
for i in range(1, 67):
   from_coordinate = landmark_tuple[i]
   to_coordinate = landmark_tuple[i+1]
   routes.append(from_coordinate)
 
from_coordinate = landmark_tuple[0]
to_coordinate = landmark_tuple[67]
routes.append(from_coordinate)
routes.append(to_coordinate) 

for i in range(18, 22):
    from_coordinate = landmark_tuple[i]
    to_coordinate = landmark_tuple[i+1]
    routes.append(from_coordinate)
 
from_coordinate = landmark_tuple[18]
to_coordinate = landmark_tuple[22]
routes.append(from_coordinate)
routes.append(to_coordinate)
  
for i in range(23, 27):
    from_coordinate = landmark_tuple[i]
    to_coordinate = landmark_tuple[i+1]
    routes.append(from_coordinate)
 
from_coordinate = landmark_tuple[23]
to_coordinate = landmark_tuple[27]
routes.append(from_coordinate)
routes.append(to_coordinate)
for i in range(28, 31):
    from_coordinate = landmark_tuple[i]
    to_coordinate = landmark_tuple[i+1]
    routes.append(from_coordinate)
 
from_coordinate = landmark_tuple[28]
to_coordinate = landmark_tuple[31]
routes.append(from_coordinate)
routes.append(to_coordinate)
 
for i in range(32, 36):
    from_coordinate = landmark_tuple[i]
    to_coordinate = landmark_tuple[i+1]
    routes.append(from_coordinate)
 
from_coordinate = landmark_tuple[32]
to_coordinate = landmark_tuple[36]
routes.append(from_coordinate)
routes.append(to_coordinate)
for i in range(37, 42):
    from_coordinate = landmark_tuple[i]
    to_coordinate = landmark_tuple[i+1]
    routes.append(from_coordinate)
 
from_coordinate = landmark_tuple[37]
to_coordinate = landmark_tuple[42]
routes.append(from_coordinate)
routes.append(to_coordinate)
 
for i in range(43, 48):
    from_coordinate = landmark_tuple[i]
    to_coordinate = landmark_tuple[i+1]
    routes.append(from_coordinate)
 
from_coordinate = landmark_tuple[43]
to_coordinate = landmark_tuple[48]
routes.append(from_coordinate)
routes.append(to_coordinate)

for i in range(49, 67):
    from_coordinate = landmark_tuple[i]
    to_coordinate = landmark_tuple[i+1]
    routes.append(from_coordinate)
 
from_coordinate = landmark_tuple[49]
to_coordinate = landmark_tuple[67]
routes.append(from_coordinate)
routes.append(to_coordinate)



mask = np.zeros((img.shape[0], img.shape[1]))
mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
mask = mask.astype(np.bool)
 
out = np.zeros_like(img)
out[mask] = img[mask]
 
matplotlib.pyplot.imshow(out)