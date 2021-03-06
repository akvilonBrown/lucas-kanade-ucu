# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 18:42:29 2020

@author: Yaroslav
"""

import numpy as np
import cv2
import utils as ut
import matplotlib.pyplot as plt

if __name__ == '__main__':
     frame = 1
     frame_str = str(frame).zfill(4) 
     #folder_path = './ClifBar/img/'
     folder_path = './Toy/img/'
     folder_path = './CarScale/img/'
     img_file = folder_path + frame_str + '.jpg'
     template = cv2.imread(img_file, 0)
     height, width = template.shape
     #template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
     
     roi = np.array([[152, 102], [192, 169]], np.int)   #toy
     roi = np.array([[6,166],	[6+42,166+26]], np.int) #CarScale 
     roi_points = ut.rect_corners(roi)
     #(140,	117) (175, 181)
     #roi = np.array([[140,	117], [175, 181]], np.int) #clifBar
     
     frame = 2
     
     fourcc = cv2.VideoWriter_fourcc(*'XVID')
     out = cv2.VideoWriter("track_Car.avi", fourcc, 10.0, (width,height))
     
     templ_out = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
     templ_out = cv2.rectangle(templ_out, tuple(roi[0]), tuple(roi[1]), (0,250, 0), 2, cv2.LINE_AA)
     #templ_out = draw_rect(templ_out, roi_points)
     plt.imshow(templ_out)
     plt.show()   
     
     while frame < 120:
         img_file = folder_path + frame_str + '.jpg'
         image = cv2.imread(img_file,0)
         
         if image is None:
             print('No Image found')
             break
     
     
         roi_new, roi_points_new = ut.lk_wrapper_v2(image, template, roi, roi_points)
     
         #templ_out = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
         #templ_out = cv2.rectangle(templ_out, tuple(roi[0]), tuple(roi[1]), (0,250, 0), 2, cv2.LINE_AA)
         #templ_out = draw_rect(templ_out, roi_points)
         
         image_out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
         image_out = cv2.rectangle(image_out, tuple(roi_new[0]), tuple(roi_new[1]), (0,250, 0), 2, cv2.LINE_AA)
         image_out = ut.draw_rect(image_out, roi_points_new)
         
         
         plt.imshow(image_out)
         plt.show() 
         
         frame += 1
         frame_str = str(frame).zfill(4)
         
         
         roi = roi_new 
         roi_points = roi_points_new
         #roi_points = rect_corners(roi_new)
         template = image.copy()
         
         
         
         #print('frame----------------', frame)
         
         out.write(image_out)
         
         
     out.release()    
     cv2.destroyAllWindows()
