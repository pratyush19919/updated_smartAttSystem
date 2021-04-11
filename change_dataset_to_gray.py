import cv2

i=1
while i<201:
    img=cv2.imread("Images/Achal3/"+str(i)+".jpg")
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    cv2.imwrite("Images/Achal3/"+str(i)+".jpg",gray)
    i=i+1