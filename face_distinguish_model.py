import cv2 
import os
import numpy as np

def face_detection(test_img):
    gray_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_c = cv2.CascadeClassifier(r'C:\.....\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
    
    face = face_c.detectMultiScale(gray_img,2,4)
    return face,gray_img


def labels_for_training_data(directory):
    faces = []
    faceID = []
    
    for path, subdirname, filename in os.walk(directory):
        for i in filename:
            if i.startswith('.'):
                print('skipping system file')
                
                continue
            
            id = os.path.basename(path)
            print('id: ',id)
            img_path = os.path.join(path,i)
            print('img_path: ',img_path)
            
            test_img= cv2.imread(img_path)
            if test_img is None:
                print('No image loaded')
                continue
            
            face,gray_img = face_detection(test_img)
            if len(face)!=1:
                print('give single face')
                continue
            (x,y,w,h)=face[0]
            roi_gray = gray_img[y:y+w, x:x+h]
            
            faces.append(roi_gray)
            faceID.append(int(id))
            
    return faces,faceID
    
def train_classifier(faces,faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer

def draw_rect(test_img,face):
    (x,y,w,h) = face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,255),1)
    
            
def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),4)            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    