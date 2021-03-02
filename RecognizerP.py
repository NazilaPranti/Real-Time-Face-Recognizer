#trainer.py
#DatasetCreator.py
#Detector.py

import cv2,os
import numpy as np
from PIL import Image
from num2words import num2words
from subprocess import call
import sqlite3


#recognizer = cv2.face_LBPHFaceRecognizer.create();
path='dataSet'

def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces = [];
    IDs = [];   
    for imagePath in imagePaths:
         if(imagePath!='dataSet\\Thumbs.db'):
             faceImg = Image.open(imagePath).convert('L');
             faceNp = np.array(faceImg, 'uint8')
             ID=int(os.path.split(imagePath)[-1].split('.')[1])
             faces.append(faceNp);
             #print ID
             IDs.append(ID)
             cv2.imshow("training", faceNp)
             cv2.waitKey(10)
    return IDs , faces

def train():  
    Ids,faces = getImagesWithID(path)
    rec.train(faces,np.array(Ids))
    rec.save('recognizer/trainingData.yml');
    #cv2.destroyAllWindows()


def insertOrUpdate(Id,Name):
    conn=sqlite3.connect("Facebase.db")
    cmd="SELECT * FROM People WHERE ID="+str(Id)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
         isRecordExist=1
    if( isRecordExist==1):
        cmd="UPDATE People SET Name='"+str(Name)+"' WHERE ID="+str(Id);
    else:
        cmd="INSERT INTO People(ID,NAME) Values("+str(Id)+","+str(Name)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close();


#id=raw_input('Enter User Id : ')
#name=raw_input('Enter User Name : ')
#insertOrUpdate(id,name)
#sampleNum=0;

def dataCreate():
    cmd = 'Please_enter_your_ID_in_Py_Shell';
    call([cmd_beg+cmd+cmd_end], shell=True);
    id = raw_input('Enter your id: ');
    if(id=='0'):
        return;
    cmd = 'Please_enter_your_name';
    call([cmd_beg+cmd+cmd_end], shell=True);
    name = raw_input('Enter your Name: ');
    insertOrUpdate(id,name);
    sampleNum=0;

    while(True):
        ret,img = cam.read();
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=faceDetect.detectMultiScale(gray,1.3,5);
        for(x,y,w,h) in faces:
            sampleNum=sampleNum+1;
            cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.waitKey(100);
        cv2.imshow("Face",img);
        cv2.waitKey(1);
        if(sampleNum>20):
            break;
    train();

def getProfile(id):
     conn=sqlite3.connect("Facebase.db")
     cmd="SELECT * FROM People WHERE ID="+str(id)
     cursor=conn.execute(cmd)
     profile=None
     for row in cursor:
         profile=row
     conn.close()
     return profile


cmd_beg= 'espeak '
cmd_end= ' 2>/dev/null' # To dump the std errors to /dev/null


#detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceDetect =cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam = cv2.VideoCapture(0)
#cam.set(3,640);
rec = cv2.face_LBPHFaceRecognizer.create();
train();
rec.read("recognizer\\trainingData.yml")

id = 0;
c=0
prename="";
font = cv2.FONT_HERSHEY_SIMPLEX
name="";
age="";
gender="";


while(True):
    ret,img = cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(conf<70):
            c=0;
            profile=getProfile(id)      
            if(profile!=None):
                name=str(profile[1]);
                age= str(profile[2]);
                gender=str(profile[3]);
                #cv2.putText(img,str(profile[1]),(x,y+h+30),font,1,(0,0,255),2);
                #cv2.putText(img,str(profile[2]),(x,y+h+60),font,1,(0,0,255),2);
                #cv2.putText(img,str(profile[3]),(x,y+h+90),font,1,(0,0,255),2);
        else:
            name="unknown"
            c=c+1;
            if(c>20):
                cmd = 'Welcome_'+name;
                call([cmd_beg+cmd+cmd_end], shell=True);
                dataCreate();
                c=0;
                
        cv2.putText(img,name,(x,y+h+30),font,1,(0,0,255),2);
        cv2.putText(img,age,(x,y+h+60),font,1,(0,0,255),2);
        cv2.putText(img,gender,(x,y+h+90),font,1,(0,0,255),2);
    cv2.imshow("Face", img);
    if(cv2.waitKey(1)==ord('q')):
        cam.release();
        cv2.destroyAllWindows();

