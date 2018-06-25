import cv2

cam = cv2.VideoCapture(0)
classificador = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
recognizer = cv2.createEigenFaceRecognizer()
recognizer.load('classificadorEigen.yml')
largura, altura = 220,220
font = cv2.FONT_HERSHEY_PLAIN
while(True):
    mat, img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    facesDetectadas = classificador.detectMultiScale(gray, scaleFactor=1.5,minSize=(50,50))
    for(x,y,l,a) in facesDetectadas:
        imgFace = cv2.resize(gray[y:y+a,x:x+l], (largura,altura))
        cv2.rectangle(img,(x,y),(x+l,y+a),(0,0,255),2)
        id, trust = recognizer.predict(imgFace)
        cv2.putText(img,str(id),(x,y+(a+30)),font,1,(0,255,0))


    cv2.imshow("Face reconhecida",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()