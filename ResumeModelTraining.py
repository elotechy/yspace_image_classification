# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from tensorflow.keras.models import load_model
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import keras


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to model")
ap.add_argument("-ts", "--testset", required=True,
	help="path to input test set")
args = vars(ap.parse_args())


keras_model = load_model(args["model"])
print("[INFO] Model loaded Successfully...")
# from keras.utils import plot_model
# plot_model(model, to_file = "model.png")
def getData(imagePaths):
    data = []
    labels = []
    for imagePath in imagePaths:
        #print(imagePath)
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]

        # load the image, swap color channels, and resize it to be a fixed
        # 224x224 pixels while ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)
        
    # convert the data and labels to NumPy arrays while scaling the pixel
    # intensities to the range [0, 255]
    data = np.array(data) / 255.0
    labels = np.array(labels)
    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    

    return data, labels, lb


print("[INFO] loading test images...")
Test_imagePaths = list(paths.list_images(args["testset"])) 
X_test, y_test ,lb = getData(Test_imagePaths)
print(y_test)
print("[INFO] Plotting ROC curve...")
from sklearn.metrics import roc_curve
y_pred_keras = keras_model.predict(X_test)
print(y_pred_keras)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test.argmax(axis=1), y_pred_keras.argmax(axis=1))

from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
#plt.show()
# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
#plt.show()

cm = metrics.confusion_matrix(lb.classes_, y_pred_keras.argmax(axis=1))
# or
#cm = np.array([[1401,    0],[1112, 0]])

plt.imshow(cm, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()

