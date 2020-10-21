# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 18:42:29 2020

@author: Yaroslav
"""

import numpy as np
import cv2
import utils as ut
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
     debug = True
     num_of_frames = 60
     frame = 1
     frame_str = str(frame).zfill(4) 
     #print (len(sys.argv))
     
     if (len(sys.argv) > 1):
         folder_path = sys.argv[1]
         roi = np.array([[sys.argv[2], sys.argv[3]], [sys.argv[4], sys.argv[5]]], \
                        np.int)
         if str(sys.argv[6]) == '-debug': debug = True         
         else: debug = False
         num_of_frames = int(sys.argv[7])
     else:
         # Toy dataset from http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/Toy.zip
         #folder_path = './Toy/img/'
         #folder_path = './DragonBaby/img/'
         folder_path = './CarScale/img/'
         #roi = np.array([[152, 102], [192, 169]], np.int)   #toy
         #roi = np.array([[160,83],[160+56, 83+65]], np.int) #Baby
         roi = np.array([[6,166],	[6+42,166+26]], np.int) #CarScale
    

     img_file = folder_path + frame_str + '.jpg'
     template = cv2.imread(img_file, 0)
     height, width = template.shape   

     roi_points = ut.rect_corners(roi)
     
     frame = 2
     
     fourcc = cv2.VideoWriter_fourcc(*'XVID')
     out = cv2.VideoWriter("track_Toy.avi", fourcc, 10.0, (width,height))
     out = cv2.VideoWriter("track_Baby.avi", fourcc, 10.0, (width,height))
     
     templ_out = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
     templ_out = cv2.rectangle(templ_out, tuple(roi[0]), tuple(roi[1]), (0,250, 0), 2, cv2.LINE_AA)

     if debug:
         plt.imshow(templ_out)
         plt.show()   
     
     while frame <= num_of_frames:
         img_file = folder_path + frame_str + '.jpg'
         image = cv2.imread(img_file,0)
         
         if image is None:
             print('No Image found')
             break
     
     
         roi_new, roi_points_new = ut.lk_wrapper(image, template, roi, roi_points)
     
         
         image_out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
         image_out = cv2.rectangle(image_out, tuple(roi_new[0]), tuple(roi_new[1]), (0,250, 0), 2, cv2.LINE_AA)
         image_out = ut.draw_rect(image_out, roi_points_new)
         
         if debug:
             plt.imshow(image_out)
             plt.show() 
         
         frame += 1
         frame_str = str(frame).zfill(4)
         
         
         roi = roi_new 
         roi_points = roi_points_new
         template = image.copy()
         
         
         
         #print('frame----------------', frame)
         
         out.write(image_out)
         
         
     out.release()    
     cv2.destroyAllWindows()
