# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 12:25:42 2021

@author: Mahfuz_Shazol
"""
#pip install cmake
#pip install dlib==19.18.0
#visual studio C++ should be Installed
#conda install -c conda-forge dlib
#pip install face-recognition

import cv2
import numpy as np
import face_recognition as fr
import os
from  datetime import datetime as dt

originalPath='MainImages'
images=[]
classeNames=[]


myList=os.listdir(originalPath)

for cl in myList:
    imgFile=cv2.imread(f'{originalPath}/{cl}')
    images.append(imgFile)
    classeNames.append(os.path.split(cl)[1].split('.')[0])
print(classeNames)

def makeAttendance(name):
    with open('Attendance.csv','r+') as f:
         dataList=f.readlines()
         nameList=[]
         for line in dataList:
             entry=line.split(',')
             nameList.append(entry[0])
         if name not in nameList:
             now=dt.now()
             dtString=now.strftime('%H:%M:%S')
             f.writelines(f'\n{name},{dtString}')


def findEncodings(images):
    encodings=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)        
        encode=fr.face_encodings(img)[0]
        encodings.append(encode)
    return encodings

encodingKnownList=findEncodings(images)
#print(len(encodingKnownList))
 

cap=cv2.VideoCapture(0)


while True:
      success,img=cap.read()
      resizedImg=cv2.resize(img,(0,0),None,0.25,0.25)       
      resizedImg=cv2.cvtColor(resizedImg,cv2.COLOR_BGR2RGB)
      locations=fr.face_locations(resizedImg)
      encodes=fr.face_encodings(resizedImg)
      
      for encode,location in zip(encodes,locations): 
          matches=fr.compare_faces(encodingKnownList,encode)
          diff=fr.face_distance(encodingKnownList,encode)
          #print(diff)    
          matchIndex=np.argmin(diff)
          if matches[matchIndex]:
              name=classeNames[matchIndex].upper()
              makeAttendance(name)
              #print(name)
              y1,x2,y2,x1=location
              y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
              cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
              #cv2.rectangle(img,(x1,y1-35),(x2,y2),(0,0,255),cv2.FILLED)
              cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,225,255))
          
      cv2.imshow('Mahfuz Shazol',img)
      cv2.waitKey(5)     
          
      if cv2.waitKey(1) & 0xff == ord('c'):
             cap.release()
             cv2.destroyAllWindows()
             break    
         
       

        
# img=fr.load_image_file('image1.jpg')
# #img=fr.load_image_file('mahfuz_shazol.jfif')
# img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# testImg=fr.load_image_file('mahfuz_shazol.jfif')
# testImg=cv2.cvtColor(testImg,cv2.COLOR_BGR2RGB)


# location=fr.face_locations(img)
# encodeLocation=fr.face_encodings(img)[0]


# testImgtlocation=fr.face_locations(testImg)
# testImgencodeLocation=fr.face_encodings(testImg)[0]



# if len(location)>0:
#     cv2.rectangle(img,(location[0][3],location[0][0]),(location[0][1],location[0][2]),(0,0,255),2)
    
# if len(testImgtlocation)>0:
#     cv2.rectangle(testImg,(testImgtlocation[0][3],testImgtlocation[0][0]),(testImgtlocation[0][1],testImgtlocation[0][2]),(0,0,255),2)    

# result=fr.compare_faces([encodeLocation], testImgencodeLocation)
# print(result)

# cv2.imshow('Mahfuz Shazol',img)
# cv2.waitKey(0)

