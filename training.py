#-*- coding: utf-8 -*-
import cv2, os
import numpy as np
import recognizer
from time import sleep
lbph = cv2.createLBPHFaceRecognizer()
def getImages():
    print('Treinando imagens...')
    path = [os.path.join('faces', f) for f in os.listdir('faces')]
    faces = []
    ids = []

    for paths in path:
        imgFace = cv2.cvtColor(cv2.imread(paths), cv2.COLOR_BGR2GRAY)
        ids.append(int(paths.split('.')[1]))
        faces.append(imgFace)

    return np.array(ids), faces
def train():
    ids, faces = getImages()

    try:
        lbph.train(faces, ids)
        lbph.save('classificadorLBPH.yml')
        print('Imagens treinadas com sucesso!')
        print('Reconhecendo rostos!')
        recognizer.predict()
    except cv2.error as e:
        print('Erro ao treinar imagens => '+ e.message)