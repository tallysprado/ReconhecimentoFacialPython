import cv2, os
import numpy as np

eigenFaces = cv2.createEigenFaceRecognizer()

def getImages():
    path = [os.path.join('faces', f) for f in os.listdir('faces')]
    faces = []
    ids = []

    for paths in path:
        imgFace = cv2.cvtColor(cv2.imread(paths), cv2.COLOR_BGR2GRAY)
        ids.append(int(paths.split('.')[1]))
        faces.append(imgFace)

    return np.array(ids), faces

ids, faces = getImages()

eigenFaces.train(faces, ids)
eigenFaces.save('classificadorEigen.yml')
