# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:22:53 2019

@author: jh
"""

import utils
import track_utils
import time
import cv2
import argparse
import numpy as np
import math
import copy


#screen stuff===============================
screen_width=900
screen_height=400
depth=3
strip_height=100
speed=5

bgcolor=track_utils.yellow
trackcolor=track_utils.black

canvas=track_utils.build_screen(strip_height,screen_height,screen_width,depth,bgcolor,trackcolor)
rect=track_utils.Rectangle(screen_width,screen_height,strip_height,speed,trackcolor)
#===============================================================

ap=argparse.ArgumentParser()
ap.add_argument('img',type=str)
ap.add_argument('frame_h',type=int)
ap.add_argument('frame_w',type=int)

ap.add_argument('data',type=str)

args=ap.parse_args()
img_url=args.img
frame_h=args.frame_h
frame_w=args.frame_w
data_url=args.data

numbering=False

data_cap=utils.predictionGrabber(data_url).start()
img_cap=utils.frameGrabber(img_url).start()

time.sleep(5)
original_img=img_cap.read()

window=50
j=0
old_vec=np.array([0,0])
speed=3
readings=np.array([])
while True:
    cv2.imshow('',canvas)
    j+=1
#    canvas=np.zeros((frame_h,frame_w,3))
    info=data_cap.read()
    if info !='pass':
        parts,scale=utils.parse_parts(info,original_img,canvas)

        if 4 in parts.keys():
            hand=parts[4]
            vec=np.array([hand['x'],hand['y']])
            norm=np.linalg.norm(vec-old_vec)
            readings=np.append(readings,norm)
            if len(readings)<window:
                speed=int(np.mean(readings))
                speed=max(speed,3)
            else:
                readings=np.array([])
            old_vec=copy.deepcopy(vec)
    
#    new_canvas=utils.draw_human(canvas,parts,scale)

        if rect.on_canvas:
            rect.update()
            new_canvas=rect.draw(copy.deepcopy(canvas))
            new_canvas=utils.draw_human(new_canvas,parts,scale)
            cv2.imshow('',new_canvas)
        else:
            print(speed)
            rect=track_utils.Rectangle(screen_width,screen_height,strip_height,speed,trackcolor)
    if cv2.waitKey(1) & 0xff == ord('q'):
        cv2.destroyAllWindows()
        break

    
        


#        if 2 in parts.keys() or 5 in parts.keys():
#            i+=1
#            try:
#                shoulder=parts[2]['y']
#            except:
#                shoulder=parts[5]['y']
#            shoulder_pos+=shoulder
#            shoulder_avg=shoulder_pos/i
#            if abs(shoulder-shoulder_avg)/shoulder_avg>0.3:
##                print('jumpingggggggggggggggggggggggggggg')
#                shoulder_pos-=shoulder
#        canvas=utils.draw_human(canvas,parts,scale)
#    cv2.imshow('',canvas)
#    print()


    
    
    
    
    

    


    


        
