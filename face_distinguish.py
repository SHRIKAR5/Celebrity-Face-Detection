import cv2
import os
import numpy as np
import face_distinguish as fr

#to test a random image
test_img=cv2.imread(r'C:/.../srk.jpg')
face_detected,gray_img=fr.face_detection(test_img)
print('faces_detected:',face_detected)

#comment this line when runing 2nd time as it saves training.yml as directory
faces,face_Id=fr.labels_for_training_data('photos')
face_Id
print(face_Id)
faces
face_recognizer=fr.train_classifier(faces,face_Id)
face_recognizer.write('training.yml')#to save the file

#un comment below lines for subsequent runs
#face_recognizer=cv2.face.LBPHFaceRecognizer_create()
#face_recognizer.read('trainingData.yml')#use this to load training data to run subsequent ines

name={0:"SalmanKhan",1:"ShahrukhKhan"}

for face in face_detected:
    (x,y,h,w)=face
    roi_gray=gray_img[y:y+h,x:x+w]
    label,confidance=face_recognizer.predict(roi_gray)
    print("confidance:",confidance)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predict_name=name[label]
    if(confidance<37):#if confidance is more than 37 than dont print predicted faces on screen
        continue
    fr.put_text(test_img,predict_name,x,y)
print(face_detected)  
resized_img=cv2.resize(test_img,(1000,1000))
cv2.imshow('face detection tutorial',resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
