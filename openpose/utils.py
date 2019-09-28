# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 13:29:35 2019

@author: jh
"""
from threading import Thread
import cv2
import requests
import numpy as np

def parse_info(p):
    p=p.replace('<','')
    p=p.replace('>','')
    info=p.split('|')
    img_dims=info[0].split(',')
    h=int(img_dims[0])
    w=int(img_dims[1])
    #
    del info[0]
    
    joint_info={}
    for piece in info:
        joint=piece.split(',')
        joint_id=int(joint[0])
        joint_x=int(float(joint[1])*w)
        joint_y=int(float(joint[2])*h)
        subdict={'x':joint_x,'y':joint_y}
        joint_info[joint_id]=subdict
    return joint_info,h,w

def parse_parts(incoming,original,canvas):
    original_h,original_w,_=original.shape
    canvas_h,canvas_w,_=canvas.shape
    
    scale_h=canvas_h/original_h
    scale_w=canvas_w/original_w
    scale=min(scale_h,scale_w)
    
    parts={}
    info=incoming.split('|')
    for message in info:
        data=message.split(',')
        part_id=int(data[0])
        part_x=int(float(data[1])*original_w*scale)
        part_y=int(float(data[2])*original_h*scale)
        part_score=float(data[3])
        subdict={'x':part_x,'y':part_y,'score':part_score}
        parts[part_id]=subdict
    
    return parts,scale


def draw_human(img,parts,scale,numbering=False):
    connections=[(0,1),(1,2),(1,5),(1,8),(1,11),(2,3),(3,4),(5,6),(6,7),(8,9),(9,10),(11,12),(12,13)]
    keys=list(parts.keys())
#    print(keys)
    thickness=4
    line_color=(0,255,0)
    joint_color=(0,255,255)
    radius=int(2)
    
    for i in parts.keys():
        if i<14:
            joint=parts[i]
            joint_x=int(joint['x'])
            joint_y=int(joint['y'])
            cv2.circle(img, (joint_x,joint_y), radius , joint_color, -1)
            if numbering:
                cv2.putText(img,str(i), (joint_x,joint_y),cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2)


    for pair in connections:
#        print(pair)
        start=pair[0]
        end=pair[1]
        if start in keys and end in keys:
            x1=int(parts[start]['x'])
            x2=int(parts[end]['x'])
            y1=int(parts[start]['y'])
            y2=int(parts[end]['y'])
            cv2.line(img, (x1, y1), (x2, y2), line_color , thickness)

    return img


class frameGrabber:
    
    def __init__(self,url):
        self.r=requests.get(url,stream=True)
        self.img=None
        self.stopped=False
        print(self.r.status_code)       
                  
        
    def start(self):
        print('starting thread')
        t=Thread(target=self.update,args=())
        t.setDaemon(True)
        t.start()
        print('thread started')
        return self
       
    def update(self):
        while True:
            if self.stopped==True:
                return
            else:
                img_bytes=bytes()
                for chunk in self.r.iter_content(chunk_size=1024):
                    img_bytes+=chunk
                    a=img_bytes.find(b'\xff\xd8')
                    b=img_bytes.find(b'\xff\xd9')
                    if a!=-1 and b!=-1:
                        jpg=img_bytes[a:b+2]
                        img_bytes=img_bytes[b+2:]
                        self.img=cv2.imdecode(np.frombuffer(jpg,dtype=np.uint8),cv2.IMREAD_COLOR)
    def read(self):
        return self.img
    
    def stop(self):
        self.stopped=True
        
class predictionGrabber(frameGrabber):
          
    def update(self):
        while True:
            if self.stopped==True:
                return
            else:
                msg=bytes()
                for chunk in self.r.iter_content():
                    msg+=chunk
                    a=msg.find(b'<')
                    b=msg.find(b'>')
                    if a!=-1 and b!=-1:
                        val=msg[a+1:b]
                        msg=msg[b+1:]
                        self.img=val.decode()



