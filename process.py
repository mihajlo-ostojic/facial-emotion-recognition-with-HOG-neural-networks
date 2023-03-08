# import libraries here
import os
import numpy as np
import cv2  # OpenCV
from sklearn.svm import SVC  # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # KNN
from joblib import dump, load
import matplotlib
import matplotlib.pyplot as plt
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
def load_color_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def display_image(image):
    plt.imshow(image, 'gray')

def reshape_data(input_data):
    nsamples, nx, ny = input_data.shape
    return input_data.reshape((nsamples, nx*ny))

def count_emotions(emotion_list, chosen_emotion):
    number_of_emotion = 0
    for emotion in emotion_list:
        if emotion == chosen_emotion:
            number_of_emotion += 1
    return  number_of_emotion

def getHog():
    nbins = 9  # broj binova
    cell_size = (10, 10)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku
    hog = cv2.HOGDescriptor(_winSize=(200 // cell_size[1] * cell_size[1],
                                      200 // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)
    return hog

def get_region(gray_image,x,y,w,h):

    try:#ideja: povecati region oko lica jer nekad ne uzima lice u potpunosti
        region = gray_image[y-20:y + h+20, x-20:x+20 + w]#y-10:y + h+10, x-10:x+10 + w
        nebitno = cv2.resize(region, (200,200), interpolation=cv2.INTER_AREA)
        return region
    except:
        #print("Greska*********************************************")
        pass

    return gray_image[y:y + h, x:x + w]

#KOD JE U POTPUNOSTI PREUZET SA VEZBI
def train_or_load_facial_expression_recognition_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija), liste
    labela za svaku fotografiju iz prethodne liste, kao i putanju do foldera u koji treba sacuvati model nakon sto se
    istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran, ili ga samo ucitati iz foldera za serijalizaciju
    clf_svm = None
    # try:
    #     clf_svm = load('test20i3i10.joblib')
    # except:
    #     pass
    # if clf_svm is not None:
    #     print("nasao svm")
    #     return clf_svm
    # inicijalizaclija dlib detektora (HOG)
    detector = dlib.get_frontal_face_detector()
    # ucitavanje pretreniranog modela za prepoznavanje karakteristicnih tacaka
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    hog = getHog()
    should_print = False

    all_images = [] #sve sive slike
    y_train = [] #sve labele
    # for path in train_image_paths:
    #     all_images.append(load_image(path))
    for emotion in train_image_labels:
        y_train.append(emotion)
        y_train.append(emotion)
    #print("Broj neutral : ",count_emotions(y_train,"neutral"))
    x_train = []
    #y_train = y_train[0:100]
    for j in range(len(train_image_paths)):
        gray_image = load_image(train_image_paths[j])
        #gray_with_hog = hog.compute(gray_image)
        if should_print:
            print("Trenutno analizira sliku ",j)
        #if j==100:
            #break
        # detekcija svih lica na grayscale slici
        #image = load_color_image(train_image_paths[1])
        rects = detector(gray_image, 1)
        #print("tip ",type(rects))
        #print("Shape", rects.shape) nema shape
        region_for_svm = None

        # iteriramo kroz sve detekcije korak 1.
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            # odredjivanje kljucnih tacaka - korak 2
            shape = predictor(gray_image, rect)

            # shape predstavlja 68 koordinata
            shape = face_utils.shape_to_np(shape)  # konverzija u NumPy niz
            # print("Dimenzije prediktor matrice: {0}".format(shape.shape))  # 68 tacaka (x,y)
            # print("Prva 3 elementa matrice")
            # print(shape[:3])


            # konvertovanje pravougaonika u bounding box koorinate
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # crtanje pravougaonika oko detektovanog lica
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            region_for_svm = get_region(gray_image,x,y,w,h)
            #region_for_svm = gray_image[y:y+h,x:x+w]
            #test = gray_image[y:y+h,x:x+w]
            #region_for_svm2 = region_for_svm[y:y+h,x+w:x]
            #print("Velicina lica ",region_for_svm.shape)
            # plt.figure()
            # plt.imshow(test, 'gray')
            # plt.show()
            # plt.figure()
            # plt.imshow(region_for_svm,'gray')
            # plt.show()
            region_for_svm = cv2.resize(region_for_svm, (200,200), interpolation=cv2.INTER_AREA)

            #Ideja sa vezbi, asistent pricao o flipovanju slike tako da svako lice dupliramo
            region_for_svm2 = cv2.flip(region_for_svm, 1)
            # plt.figure()
            # plt.imshow(region_for_svm2, 'gray')
            # plt.show()
            #print("Velicina lica ",region_for_svm.shape)
            #print("Hog je ",len(hog.compute(region_for_svm))," i jos ",type(hog.compute(region_for_svm)))
            x_train.append(hog.compute(region_for_svm))
            x_train.append(hog.compute(region_for_svm2))
            # plt.figure()
            # plt.imshow(hog.compute(region_for_svm))
            # plt.show()
            #x_train.append(region_for_svm)
            # plt.figure()
            # plt.imshow(region_for_svm,'gray')
            # plt.show()

            # ispis rednog broja detektovanog lica
            # cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #
            # # crtanje kljucnih tacaka
            # for (x, y) in shape:
            #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    if should_print:
        print("Pre arraya")
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    if should_print:
        print('Train shape je: ', x_train.shape, y_train.shape)
    x_train = reshape_data(x_train)
    if should_print:
        print('Reshape je: ', x_train.shape, y_train.shape)
    clf_svm = SVC(kernel='linear', probability=True)
    clf_svm.fit(x_train, y_train)
    # clf_knn = KNeighborsClassifier(n_neighbors=7) #radi lose
    # clf_knn = clf_knn.fit(x_train, y_train)
    if should_print:
        print("Model obucen")
    #dump(clf_svm, 'test20i3i10.joblib')
    #model = clf_knn ovo lose radi
    model =clf_svm
    return model

#KOD JE U POTPUNOSTI PREUZET SA VEZBI
def extract_facial_expression_from_image(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje ekspresije lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati ekspresiju.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <String>  Naziv prediktovane klase (moguce vrednosti su: 'anger', 'contempt', 'disgust', 'happiness', 'neutral', 'sadness', 'surprise'
    """
    # inicijalizaclija dlib detektora (HOG)
    detector = dlib.get_frontal_face_detector()
    # ucitavanje pretreniranog modela za prepoznavanje karakteristicnih tacaka
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    loaded_image = load_image(image_path)  # siva slika
    x_train = []

    hog = getHog()
    should_print = False

    # detekcija svih lica na grayscale slici
    # image = load_color_image(train_image_paths[1])
    rects = detector(loaded_image, 1)
    # print("tip ",type(rects))
    # print("Shape", rects.shape) nema shape
    region_for_svm = None

        # iteriramo kroz sve detekcije korak 1.
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        # odredjivanje kljucnih tacaka - korak 2
        shape = predictor(loaded_image, rect)

        # shape predstavlja 68 koordinata
        shape = face_utils.shape_to_np(shape)  # konverzija u NumPy niz
        # print("Dimenzije prediktor matrice: {0}".format(shape.shape))  # 68 tacaka (x,y)
        # print("Prva 3 elementa matrice")
        # print(shape[:3])

        # konvertovanje pravougaonika u bounding box koorinate
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        # crtanje pravougaonika oko detektovanog lica
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #test = loaded_image[y:y + h, x:x + w]
        region_for_svm = get_region(loaded_image,x, y, w, h)

        # plt.figure()
        # plt.imshow(test, 'gray')
        # plt.show()
        # plt.figure()
        # plt.imshow(region_for_svm, 'gray')
        # plt.show()
        #region_for_svm = loaded_image[y:y + h, x:x + w]
        # print("Velicina lica ",region_for_svm.shape)

        region_for_svm = cv2.resize(region_for_svm, (200, 200), interpolation=cv2.INTER_AREA)
        # print("Velicina lica ",region_for_svm.shape)
        x_train.append(hog.compute(region_for_svm))
        #x_train.append(region_for_svm)


        # ispis rednog broja detektovanog lica
        # cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #
        # # crtanje kljucnih tacaka
        # for (x, y) in shape:
        #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)


    x_train = np.array(x_train)
    #print('Train shape je: ', x_train.shape)
    x_train = reshape_data(x_train)
    #print('Reshape je: ', x_train.shape)


    y_train_pred = trained_model.predict(x_train)
    #print("Nesto od predicta ",y_train_pred[0]);
    facial_expression = y_train_pred[0]
    if should_print:
        print(facial_expression)
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)

    return facial_expression
