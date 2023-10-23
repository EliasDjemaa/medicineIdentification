#importing
import customtkinter
from paddleocr import PaddleOCR, draw_ocr #importing main paddle ocr class to instantiate model and draw ocr method to visualise results
from matplotlib import pyplot as plt #importing pyplot to visualise image data
from PIL import Image
import cv2
import os #folder directory navigation


#directory for frames to be saved of medication packaging
directory = r'C:\Users\Elias\PycharmProjects\medOCR\preCF'
#initation interface with camera
cap = cv2.VideoCapture(0)
i = 0
#live loop of webcam frames until cap is closed
while cap.isOpened():
    ret, frame = cap.read()

    #show live feed of camera
    cv2.imshow('webcam', frame)

    # if statement checks if c is pressed to capture frame
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite("f" + str(i) + ".jpg", frame)
        medData_path = ("f" + str(i) + ".jpg")
        i = i + 1
        break

    #if statement checks if q is pressed to stop live feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#instantiating model and detect
#MODEL SETUP
ocr = PaddleOCR(use_angle_cls=True, lang='en') #selecting language
#medData_path = os.path.join('.', 'medData', 'training', 'shelcal', 'images5045.jpg') #path of images of the medication packaging

result = ocr.ocr(medData_path, cls=True)

#printing the list of results
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line[1][0])

#storing detected package information locally
result = result[0]
boxes = [line[0] for line in result] #cordiantes of where we can make boxes to mark text found
text = [line[1][0] for line in result] #text result
scores = [line[1][1] for line in result] #metric used to see accuracy

#outlining font for drawOCR
fPath = os.path.join('.', 'fonts', 'latin.ttf')
#importing images of medication boxes
img = cv2.imread(medData_path)#importing image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#visualisation of medication data
plt.figure(figsize=(15,15))
visu=draw_ocr(img,boxes,text,font_path=fPath)
visu = Image.fromarray(visu)
visu.save('result.jpg')
plt.imshow(visu)
plt.show()

