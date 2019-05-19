from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn import metrics
import numpy as np
import pickle
import cv2

def classify_image(model,lb,image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image,(96,96))
    image = image.astype("float")/255.0
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]
    return label

def test(test_path,ground_truth_path):
    print("Beginning test ...")
    correct = 0
    total  = 0
    ground_truth = open(ground_truth_path)
    model = load_model("modVgg.model")
    lb = pickle.loads(open("lb.pickle", "rb").read())
    ground_truth_lines = ground_truth.readlines()
    labels = []
    image_paths = []
    predictions = []
    for line in ground_truth_lines:
        splitted = line.split("\t")
        labels.append(splitted[1].replace("\n",""))
        image_paths.append(splitted[0])
    for imagePath in image_paths:
        label = labels[total]
        prediction = classify_image(model,lb,"test/" + imagePath)
        predictions.append(prediction)
        if prediction == label:
            correct += 1
        total += 1
    y_true = labels
    y_pred = predictions
    return correct / total, y_true,y_pred
def get_metrics(y_true,y_pred):
    accuracy = metrics.accuracy_score(y_true,y_pred)
    precision = metrics.precision_score(y_true,y_pred,average="weighted")
    recall = metrics.recall_score(y_true,y_pred,average="weighted")
    f1 = metrics.f1_score(y_true,y_pred,average="weighted")
    return  accuracy,precision,recall,f1


