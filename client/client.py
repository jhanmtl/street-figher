# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:22:53 2019

@author: jh
"""

import utils
import time
import cv2
import argparse

ap=argparse.ArgumentParser()
ap.add_argument('img',type=str)
ap.add_argument('scale',type=float)
ap.add_argument('data',type=str)

args=ap.parse_args()
img_url=args.img
data_url=args.data
scale=args.scale

numbering=False

data_cap=utils.predictionGrabber(data_url).start()
img_cap=utils.frameGrabber(img_url).start()
time.sleep(5)

while True:
    img=img_cap.read()
    h,w,_=img.shape
    img=cv2.resize(img,(int(w*scale),int(h*scale)))
    info=data_cap.read()
    if info !='pass':
#        print(info)
        img=utils.draw_human(img,info,numbering)
    img=cv2.flip(img,1)
    cv2.imshow('',img)
    if cv2.waitKey(1)==ord('q'):
        cv2.destroyAllWindows()
        data_cap.stop()
        img_cap.stop()
        break
    
#    print(info)
        





    
    
    
    
    

    


    


        
