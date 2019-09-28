# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 14:33:07 2019

@author: jh
"""
import utils
import time
import cv2
import argparse
import numpy as np




ap=argparse.ArgumentParser()
ap.add_argument('img',type=str)
ap.add_argument('data',type=str)

args=ap.parse_args()
img_url=args.img
data_url=args.data

numbering=False

data_cap=utils.predictionGrabber(data_url).start()
img_cap=utils.frameGrabber(img_url).start()

time.sleep(5)
original_img=img_cap.read()
h,w,d=original_img.shape
canvas=np.zeros((h,w,d))

while True:
    cv2.imshow('',canvas)
    info=data_cap.read()
    if info !='pass':
        persons=info.split('^')
        p1=persons[0]
        p2=persons[1]
        
        joint_vec1,h1,w1=utils.parse_info(p1)
        joint_vec2,h2,w2=utils.parse_info(p2)

        new_canvas=utils.draw_human(canvas,joint_vec1)
        new_canvas=utils.draw_humans(new_canvas,joint_vec2)
    cv2.imshow('',new_canvas)
    if cv2.waitKey(1) & 0xff == ord('q'):
        cv2.destroyAllWindows()
        break

    
        



    
    
    
    
    

    


    


        


