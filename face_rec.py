# facerec.py
import cv2, sys, numpy as np, os, keras
# import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import glob
size = 4
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'att_faces'
FRmodel = keras.models.load_model('facenet_keras.h5')
haar_cascade = cv2.CascadeClassifier(fn_haar)
thresh = 5

def img_to_encoding(image, model):
    image = cv2.resize(image, (160, 160))
    # img = image[..., ::-1]
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict(x_train)
    return embedding


def recognize_face(face_descriptor, database, FRmodel):
    encoding = img_to_encoding(face_descriptor, FRmodel)
    min_dist = 100
    identity = None

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding)
        similarity = cosine_similarity(db_enc,encoding)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        if dist < min_dist:
            min_dist = dist
            identity = name
            sim = similarity
    # if min_dist < 1:
    #     return identity, min_dist
    # else:
    #     return 'unknown', min_dist
    return identity,min_dist, sim[0][0]


def img_path_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    return img_to_encoding(img1, model)

def initialize():
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("database/*"):
        for x in glob.glob(file + "/*.png"):
            identity = x.split('\\')[1]
            database[identity] = img_path_to_encoding(x, FRmodel)

    return database

def extract_face_info(img,database):
    mini = cv2.resize(img, (int(img.shape[1] / size), int(img.shape[0] / size)))
    faces = haar_cascade.detectMultiScale(mini)
    faces = sorted(faces, key=lambda x: x[3])
    x, y, w, h = 0, 0, 0, 0
    if len(faces) > 0:
        for face in faces:
            (x, y, w, h) = [v * size for v in face]
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            image = img[y:y + h, x:x + w]
            name, min_dist,similarity = recognize_face(image, database,FRmodel)
            if similarity > 0.8:
                cv2.putText(img, "Face : " + name, (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                cv2.putText(img, "Similarity : " + str(similarity), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                # cv2.putText(img, "Similarity : " + str(similarity), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            else:
                cv2.putText(img, 'No matching faces', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)



def recognize():
    database = initialize()
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mini = cv2.resize(gray, (int(gray.shape[1] / size), int(gray.shape[0] / size)))
        faces = haar_cascade.detectMultiScale(mini)
        subjects = sorted(faces, key=lambda x: x[3])
        for subject in subjects:
            extract_face_info(img,database)
            (x, y, w, h) = [v * size for v in subject]
            face = img[y:y + h, x:x + w]
            cv2.imshow('captured face',face)
        cv2.imshow('Recognizing faces', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

recognize()
