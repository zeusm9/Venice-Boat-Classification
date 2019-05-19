from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
import matplotlib.pyplot as plt
from smallervggnet import SmallerVGGNet
from alexnet import Alexnet
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os

EPOCHS = 45
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96,96,3)

def train(arc):
    print("Loading images...")
    imagePaths = sorted(list(paths.list_images("train")))
    ground_truth = open("new_ground_truth.txt")
    random.seed(42)
    random.shuffle(imagePaths)

    data = []
    labels = []
    data_test = []
    data_labels = []

    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image,(IMAGE_DIMS[1],IMAGE_DIMS[0]))
        image = img_to_array(image)
        data.append(image)
        l  = imagePath.split(os.path.sep)[-2].split("_")[0].replace('\n','')
        labels.append(l)

    data = np.array(data,dtype="float") / 255.0
    labels = np.array(labels)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    test_images = os.listdir("test")

    for line in ground_truth.readlines():
        image_path = line.split("\t")[0]
        if image_path in test_images:
            image = cv2.imread("test/" + image_path)
            image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
            image = img_to_array(image)
            data_test.append(image)
            l = label = line.split("\t")[1].replace("\n", "")
            data_labels.append(l)

    data_test = np.array(data_test, dtype="float") / 255.0
    data_labels = np.array(data_labels)
    lb2 = LabelBinarizer()
    data_labels = lb2.fit_transform(data_labels)



    trainX = data
    trainY = labels
    testX = data_test
    testY = data_labels
    #(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
    aug = ImageDataGenerator(rotation_range=25,width_shift_range=0.1,height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,
                          horizontal_flip=True,fill_mode="nearest")

    if arc == "vgg":
        model = SmallerVGGNet.build(width=IMAGE_DIMS[1],height=IMAGE_DIMS[0],depth=IMAGE_DIMS[2],classes=len(lb.classes_))
    else:
        model = Alexnet.build(width=IMAGE_DIMS[1],height=IMAGE_DIMS[0],depth=IMAGE_DIMS[2],classes=len(lb.classes_))

    opt = Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)

    plot_model(model, to_file='model.png')
    model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

    print("Training the network...")

    H = model.fit_generator(
        aug.flow(trainX,trainY,batch_size=BS),
        validation_data=(testX,testY),
        steps_per_epoch=len(trainX)//BS,
        epochs=EPOCHS,verbose=1)

    if arc == "vgg":
        model.save("modVgg.model")
    else:
        model.save("modAlex.model")

    f = open("lb.pickle","wb")
    f.write(pickle.dumps(lb))
    f.close()

    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig("plot1.png")
