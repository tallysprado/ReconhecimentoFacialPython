#-*- coding: utf-8 -*-
#author: Tallys Prado. Quem fez o código, mais da metade da apresentação e do artigo

import cv2
import numpy as np


classificador = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
classificadorOlho = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
amostra = 1
id = raw_input('Digite seu ID (inteiro):')
largura, altura = 220,220
cam = cv2.VideoCapture(0)

while(True):
    mat, img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = classificador.detectMultiScale(gray,scaleFactor=1.5,minSize=(70,70))

    for (x,y,l,a) in faces:
        cv2.rectangle(img,(x,y),(x+l,y+a),(0,0,255),2)
        regiaoFace = img[y:y+a,x:x+l]
        regiaoCinzaOlho = cv2.cvtColor(regiaoFace, cv2.COLOR_BGR2GRAY)
        olhos = classificadorOlho.detectMultiScale(regiaoCinzaOlho, scaleFactor=1.4,minNeighbors=5,minSize=(30,30))

        for (ox,oy,ol,oa) in olhos:
            cv2.rectangle(regiaoFace, (ox,oy),(ox+ol, oy+oa), (0,255,0),2)

            if (cv2.waitKey(1) & 0xFF == ord('d')):
                if np.average(gray)> 60:
                    imgFace = cv2.resize(gray[y:y+a,x:x+l],(largura,altura))
                    cv2.imwrite('faces/pessoa.'+str(id)+"."+str(amostra)+'.jpg',imgFace)
                    print('pessoa.'+str(id)+str(amostra)+'.jpg salva na pasta faces')
                    amostra += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if (amostra > 25):
        break
    cv2.imshow("Webcam 0", img)

cv2.destroyAllWindows()
cam.release()