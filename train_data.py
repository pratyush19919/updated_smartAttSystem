import os
from PIL import Image
import cv2
import numpy as np
faces_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

def get_faces_id(path):
    i=1
    count=0
    dirpath=[]
    image_paths=[]
    
    for folder in (os.listdir(path)):
        for i in range(1,201):
            pic_path = path + folder + "/" + str(i) +".jpg"
            image_paths.append(pic_path)

    ids=[]
    faceSamples=[]
    # print("-----------------------------------------------------")
    # print(image_paths)
    # print("-----------------------------------------------------")
    # print(len(image_paths))
    # print("----------------------------------------------------")
    for images in image_paths:
        img=Image.open(images)
        arr=np.array(img,'uint8')
        id = images.split("/")[-2][-1]
   
        faces=faces_cascade.detectMultiScale(arr)
        if len(faces) == 0:
            print("----------------------------------------------------")
            print("OOPS!!", images)
            print("----------------------------------------------------")
        for (x,y,w,h) in faces:
            faceSamples.append(arr[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples, ids

# faces,idss = get_faces_id("B:\\NTPEL\\udemypythonprograms\\opencv\\face_recog\\train_data")
# for i in range(0, len(idss)): 
#     idss[i] = int(idss[i])
# print(idss)
# print(len(idss))
# print(len(faces))
# print("----------------------------------------------------")
# print(idss)
# print("----------------------------------------------------")
# recognizer.train(faces,np.array(idss))
# recognizer.write("B:\\NTPEL\\udemypythonprograms\\opencv\\face_recog\\trained_data\\trainer.yml")

# print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(idss))))