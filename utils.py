# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 18:42:29 2020

@author: Yaroslav
"""

import numpy as np
import cv2
from scipy.interpolate import griddata

# function to get all four points
def roi_points(roi):
  tl = np.array([roi[0,0], roi[0,1], 1], np.int)
  tr = np.array([roi[1,0], roi[0,1], 1], np.int)
  br = np.array([roi[1,0], roi[1,1], 1], np.int)
  bl = np.array([roi[0,0], roi[1,1], 1], np.int)
  return tl, tr, br, bl

# function to get all four points
def rect_corners(rect):
  assert rect.shape[0] == 2, "There should be two points only" 
  arr =  np.array([[rect[0,0], rect[0,1]],
                   [rect[1,0], rect[0,1]],
                   [rect[1,0], rect[1,1]],
                   [rect[0,0], rect[1,1]]], np.int)
  return arr 

# function to get 2 points for drawing rectangle
def get_rect(points):  
  tl_x = min(points[:,0])
  tl_y = min(points[:,1])
  br_x = max(points[:,0])
  br_y = max(points[:,1])
  return np.array([[tl_x, tl_y], [br_x, br_y]], np.int)

# function to translate coordinates from the initial image range to the smaller 
# region (assuming the first entry in the bigger_rectangle is a starting point
# either it's 2-point array or 4-point)
# this is roi box with respect to clipped image coordinates
def translate_cordinates(smaller_rectangle, start_x, start_y, reverse = False):
  arr = np.zeros_like(smaller_rectangle)
  direction = -1 if reverse else 1
  #start_x, start_y = bigger_rectangle[0][0], bigger_rectangle[0][1]
  arr[:,0] = smaller_rectangle[:,0] - start_x * direction
  arr[:,1] = smaller_rectangle[:,1] - start_y * direction
  return arr

# function producing the coordinates of area around roi
def clipping_region(roi, img_height, img_width, pad_factor):
  bb_width = roi[1][0] - roi[0][0]
  bb_height = roi[1][1] - roi[0][1]
  roi_clip_tx = max(0, roi[0][0] - int (bb_width * 0.33))
  roi_clip_ty = max(0, roi[0][1] - int (bb_height * 0.33)) 
  roi_clip_bx = min(img_width, roi[1,0] + int (bb_width * 0.33))
  roi_clip_by = min(img_height, roi[1,1] + int (bb_height * 0.33)) 

  roi_clip = np.array([[roi_clip_tx, roi_clip_ty], [roi_clip_bx, roi_clip_by]], np.int)
  return roi_clip

# function adding ones to existing coordinates in order to prepare 
# for multiplication with affine transformation matrix
def expand_with_ones(points):
  ones_arr = np.ones((points.shape[0], 1), np.int)
  prep = np.hstack((points, ones_arr))
  return prep

# 
def jacobian(x_shape, y_shape):
    
    x = np.array(range(x_shape))
    y = np.array(range(y_shape))
    x, y = np.meshgrid(x, y) 
    ones = np.ones((y_shape, x_shape))
    zeros = np.zeros((y_shape, x_shape))

    row1 = np.stack((x, zeros, y, zeros, ones, zeros), axis=2)
    row2 = np.stack((zeros, x, zeros, y, zeros, ones), axis=2)
    jacob = np.stack((row1, row2), axis=2) 

    jacob[:,:, 0, 4] = 1
    jacob[:,:, 1, 5] = 1
    return jacob

# version with ranges for usage with custom warper
def jacobian_range(x_min, x_max, y_min, y_max):
    # get jacobian of the template size.
    x = np.array(range(x_min, x_max))
    y = np.array(range(y_min, y_max))
    x_shape = x.shape[0]
    y_shape = y.shape[0]

    x, y = np.meshgrid(x, y) 
    ones = np.ones((y_shape, x_shape))
    zeros = np.zeros((y_shape, x_shape))

    row1 = np.stack((x, zeros, y, zeros, ones, zeros), axis=2)
    row2 = np.stack((zeros, x, zeros, y, zeros, ones), axis=2)
    jacob = np.stack((row1, row2), axis=2) 
    jacob[:,:, 0, 4] = 1
    jacob[:,:, 1, 5] = 1

    return jacob    


# modified jacobian matrix favoring translation over 2 dof (x and y offset)
# giving less weight to scaling/rotation
# [1, 0, 1, 0, 1, 0
#  0, 1, 0, 1, 0, 1]  instead 
def jacobian_2dof(x_shape, y_shape):
    #scale_favor = 0.1

    jacobian_template = np.zeros((2,6))
    jacobian_template[0,4] = 1.0
    jacobian_template[1,5] = 1.0

    jacob = np.zeros((y_shape, x_shape, 2, 6))
    for y in range(y_shape):
        for x in range(x_shape):
            jcell = jacobian_template.copy()
            jcell[0,0],  jcell[0,2], jcell[1,1], jcell[1,3] = 1,1,1,1 #x*scale_favor,y*scale_favor,x*scale_favor,y*scale_favor
            jacob[y,x] = jcell

    return jacob 

def jacobian_simplified(x_shape, y_shape):
    jacob = np.zeros((y_shape, x_shape, 2, 6))
    jacob[:,:, 0, 0] = .01
    jacob[:,:, 0, 2] = .01
    jacob[:,:, 0, 4] = 1
    jacob[:,:, 1, 1] = .01 
    jacob[:,:, 1, 3] = .01
    jacob[:,:, 1, 5] = 1
    return jacob    

def draw_rect(image,pts):
    cv2.polylines(image,  [pts.reshape((-1, 1, 2))],  True,  (0, 0, 254),  2)   
    return image

def crop(img, roi):
    return img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]

# custom affine warping function with reference point in the center
# of the patch
def customWarp(clipped_img, wrp_mat):
    clipped_img = clipped_img.copy()
    he, wi = clipped_img.shape

    center_y, center_x = clipped_img.shape[0]//2, clipped_img.shape[1]//2
    rct = np.array([[0,0], [wi, he]], np.int)
    clipped_img_new_rect = translate_cordinates(rct, center_x, center_y)
    #print(f"clipped_img_new_rect \n{clipped_img_new_rect}")
    x_arr = np.array(range(-center_x, wi-center_x ), np.int)
    y_arr = np.array(range(-center_y, he-center_y ), np.int)
    
    x, y = np.meshgrid(x_arr, y_arr)
    n =  np.stack((x, y), axis = 2)
    
    prepared = n.reshape(he*wi,2)
    prepared = expand_with_ones(prepared)
    
    new_coord = np.dot(wrp_mat, prepared.T)
    new_coord = new_coord.T.reshape(he,wi,2).astype(np.int)
    #min_y = np.min(new_coord[:,:,1])
    #min_x = np.min(new_coord[:,:,0])
    #max_y = np.max(new_coord[:,:,1])
    #max_x = np.max(new_coord[:,:,0])
    
    range_x = range(clipped_img_new_rect[0,0],clipped_img_new_rect[1,0])
    range_y = range(clipped_img_new_rect[0,1],clipped_img_new_rect[1,1])
    
    ys = new_coord[:,:,1].reshape(-1)
    xs = new_coord[:,:,0].reshape(-1)

    values = clipped_img.reshape(-1)
    X, Y = np.meshgrid(range_x, range_y)
    
    # this is for interpolation
    Ti = griddata((xs, ys), values, (X, Y), method='cubic')
    #res = np.round(Ti).astype(np.uint8)
    return Ti


def lk_track(img, template, rect, p, threshold = 0.001, max_iter=100):
  #print(rect) 
  #img = img.copy()
  #template = template.copy()
  KERNEL_SIZE = 3
  d_p_norm = np.inf
  template = crop(template, rect)
  rows_target, cols_target = template.shape
  rows, cols = img.shape
  p_prev = p
  iter = 0
  
  # some normalizaton required to decrease grad values growing with kernel size
  grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE)  / KERNEL_SIZE
  grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE)  / KERNEL_SIZE

  while (d_p_norm >= threshold) and iter <= max_iter:
      warp_mat = np.array([[p_prev[0], p_prev[2], p_prev[4]], [p_prev[1], p_prev[3], p_prev[5]]])
      
      warp_img = cv2.warpAffine(img, warp_mat, (img.shape[1],img.shape[0]),flags=cv2.INTER_CUBIC)
      warp_img = crop(warp_img, rect)
      diff = template.astype(int) - warp_img.astype(int)
                  
      # Calculate warp gradient of image
      
      #warp the gradient  
      grad_x_warp = crop(cv2.warpAffine(grad_x, warp_mat, (img.shape[1],img.shape[0]),flags=cv2.INTER_CUBIC), rect)
      grad_y_warp = crop(cv2.warpAffine(grad_y, warp_mat, (img.shape[1],img.shape[0]),flags=cv2.INTER_CUBIC), rect)  
      
      # Calculate Jacobian 
      jacob = crop(jacobian(cols, rows), rect)

      grad = np.stack((grad_x_warp, grad_y_warp), axis=2)
      grad = np.expand_dims((grad), axis=2)
       
      #calculate steepest descent
      steepest_descents = np.matmul(grad, jacob)
      steepest_descents_trans = np.transpose(steepest_descents, (0, 1, 3, 2))
      
      # Compute Hessian matrix
      hessian_matrix = np.matmul(steepest_descents_trans, steepest_descents).sum((0,1))
      #hessian_matrix_prime = np.matmul(steepest_descents_trans, steepest_descents)
      
      # Compute steepest-gradient-descent update
      diff = diff.reshape((rows_target, cols_target, 1, 1))
      update = (steepest_descents_trans * diff).sum((0,1))

      # calculate dp and update it
      d_p = np.matmul(np.linalg.pinv(hessian_matrix), update).reshape((-1))
      
      # for some reason delta p apeared to be negative              
      p_prev -= d_p
                
      d_p_norm = np.linalg.norm(d_p)
      iter += 1  
  return p_prev


# version with custom warper
def lk_track_v2(img, template, rect, p, threshold = 0.001, max_iter=100):
  #print(rect) 
  #img = img.copy()
  #template = template.copy()
  KERNEL_SIZE = 3
  d_p_norm = np.inf
  template = crop(template, rect)
  rows_target, cols_target = template.shape
  rows, cols = img.shape
  p_prev = p
  iter = 0
  
  # some normalizaton required to decrease grad values growing with kernel size
  grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE)  / KERNEL_SIZE
  grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE)  / KERNEL_SIZE

  while (d_p_norm >= threshold) and iter <= max_iter:
      warp_mat = np.array([[p_prev[0], p_prev[2], p_prev[4]], [p_prev[1], p_prev[3], p_prev[5]]])
      
      warp_img =  np.round(customWarp(img, warp_mat))   
      #warp_img = cv2.warpAffine(img, warp_mat, (img.shape[1],img.shape[0]),flags=cv2.INTER_CUBIC)
      warp_img = crop(warp_img, rect)
      diff = template.astype(int) - warp_img.astype(int)
                  
      # Calculate warp gradient of image
      
      #warp the gradient  
      grad_x_warp = crop(customWarp(grad_x, warp_mat), rect)
      grad_y_warp = crop(customWarp(grad_y, warp_mat), rect)

      # Calculate Jacobian 
      jacob = crop(jacobian_range(-cols//2, cols-cols//2, -rows//2, rows-rows//2), rect)  

      grad = np.stack((grad_x_warp, grad_y_warp), axis=2)
      grad = np.expand_dims((grad), axis=2)
       
      #calculate steepest descent
      steepest_descents = np.matmul(grad, jacob)
      steepest_descents_trans = np.transpose(steepest_descents, (0, 1, 3, 2))
      
      # Compute Hessian matrix
      hessian_matrix = np.matmul(steepest_descents_trans, steepest_descents).sum((0,1))
      #hessian_matrix_prime = np.matmul(steepest_descents_trans, steepest_descents)
      
      # Compute steepest-gradient-descent update
      diff = diff.reshape((rows_target, cols_target, 1, 1))
      update = (steepest_descents_trans * diff).sum((0,1))

      # calculate dp and update it
      d_p = np.matmul(np.linalg.pinv(hessian_matrix), update).reshape((-1))
                    
      p_prev -= d_p
      #print(f"d_p {d_p}")
                
      d_p_norm = np.linalg.norm(d_p)
      iter += 1  
  return p_prev

def lk_wrapper(img, template, roi, roi_points):
     # this is padding area around roi
     pad_factor = 0.33

     #initial warp (no transformation = identity)
     p = [1.0, 0 , 0, 1, 0, 0]

     # We need to downsize/crop the image in order to decrease computations for
     # back warping and move reference point closer to original bounding box.
     # This should simplify transformation, rotation in particular 
     # (because it's hardly possible to rotate image with reference point 
     # at the top left corner)
     # We assume that tracked object didn't move further than 1/3 from original bounding box position
     # so let's define new clipping region 1/3 larger than roi
     clip_region = clipping_region(roi, template.shape[0], \
                                   template.shape[1], pad_factor)
     
     # obtaining smaller region instead of full size image 
     clipped_img = crop(img, clip_region).copy()
     templ = crop(template, clip_region).copy()
     start_x, start_y = clip_region[0][0], clip_region[0][1]

     # retriving roi coordinates with respect
     translated_roi = translate_cordinates(roi, start_x, start_y)
     translated_roi_corners =  translate_cordinates(roi_points, start_x, start_y)
     
     p = lk_track(clipped_img, templ, translated_roi,  p)

     warp_mat = np.array([[p[0], p[2], p[4]], [p[1], p[3], p[5]]])
     warp_mat_inv = cv2.invertAffineTransform(warp_mat)
     prepared_points = expand_with_ones(translated_roi_corners)
     warped_points = (warp_mat_inv @ prepared_points.T).astype(int)
     restored_points = translate_cordinates(warped_points.T, start_x, start_y, reverse=True)
     new_roi_rect = get_rect(restored_points)

     return new_roi_rect, restored_points 

# version with custom warper
def lk_wrapper_v2(img, template, roi, roi_points):
     # this is padding area around roi
     pad_factor = 0.33

     #initial warp (no transformation = identity)
     p = [1.0, 0 , 0, 1, 0, 0]

     clip_region = clipping_region(roi, template.shape[0], \
                                   template.shape[1], pad_factor)
     
     # obtaining smaller region instead of full size image 
     clipped_img = crop(img, clip_region).copy()
     templ = crop(template, clip_region).copy()
     start_x, start_y = clip_region[0][0], clip_region[0][1]
     clipped_height, clipped_width = clipped_img.shape
     start_x_ad, start_y_ad =  clipped_width//2, clipped_height//2
     # retriving roi coordinates with respect
     translated_roi = translate_cordinates(roi, start_x, start_y)
     translated_roi_corners =  translate_cordinates(roi_points, start_x, start_y)

     
     translated_roi_corners_ad =  translate_cordinates(translated_roi_corners, start_x_ad, start_y_ad)

     
     p = lk_track_v2(clipped_img, templ, translated_roi,  p)

     warp_mat = np.array([[p[0], p[2], p[4]], [p[1], p[3], p[5]]])
     warp_mat_inv = cv2.invertAffineTransform(warp_mat)
     
     #small correction
     if warp_mat_inv[0,0] < 1: 
         warp_mat_inv[0,0] = warp_mat_inv[0,0] *1.05
         
     if warp_mat_inv[1,1] < 1: 
         warp_mat_inv[1,1] = warp_mat_inv[1,1] *1.05

     
     prepared_points = expand_with_ones(translated_roi_corners_ad)
     warped_points = (warp_mat_inv @ prepared_points.T).astype(int)     
     restored_points = translate_cordinates(warped_points.T, start_x_ad, start_y_ad, reverse=True)
     restored_points = translate_cordinates(restored_points, start_x, start_y, reverse=True)
     new_roi_rect = get_rect(restored_points)

     return new_roi_rect, restored_points     

def resample_image(image, iteration):
    for i in range(iteration):
        image = cv2.pyrDown(image)
    return image

def lk_pyramid_wrapper(entry_image, entry_template, roi, roi_points, num_of_layers):
    
    for i in reversed(range(num_of_layers+1)):
        
        image = resample_image(entry_image, i)
        template = resample_image(entry_template, i)
        scale = 2**i
        roi = roi//scale
        roi_points = roi_points//scale
        roi_new, roi_points_new = lk_wrapper(image, template, roi, roi_points)
        roi = roi_new * scale
        roi_points = roi_points_new * scale
        
    return roi, roi_points