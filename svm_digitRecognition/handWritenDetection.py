import csv
from sklearn import svm, metrics
from numpy import genfromtxt
import numpy as np

dataset = genfromtxt('mnist_train.csv', delimiter=",", dtype=np.dtype('i4'))[1:]
labels = [x[0] for x in dataset]
data = [x[1:] for x in dataset]

n_samples = len(labels)
n_features = len(data[0])

print("Number of samples: " + str(n_samples) + ", number of features: "+ str(n_features))

# a support vector classifier
classifier = svm.LinearSVC()

split_point = int(n_samples * 0.66)

# using two thirds for training
# ans one third for testing

labels_learn = labels[:split_point]
data_learn = data[:split_point]

labels_test = labels[split_point:]
data_test = data[split_point:]

print("Training: " + str(len(labels_learn)) + " Test: " + str(len(labels_test)))

# Learning Phase
classifier.fit(data_learn, labels_learn)

# Predict Test Set
predicted = classifier.predict(data_test)

# classification report
#print("Classification report for classifier %s:n%sn" % (classifier, metrics.classification_report(labels_test, predicted)))

# confusion matrix
print("Confusion matrix:n%s" % metrics.confusion_matrix(labels_test, predicted))






# #for create .csv
# ##############
# def convert(imgf, labelf, outf, n):
#     f = open(imgf, "rb")
#     o = open(outf, "w")
#     l = open(labelf, "rb")
#
#     f.read(16)
#     l.read(8)
#     images = []
#
#     for i in range(n):
#         image = [ord(l.read(1))]
#         for j in range(28*28):
#             image.append(ord(f.read(1)))
#         images.append(image)
#
#     for image in images:
#         o.write(",".join(str(pix) for pix in image)+"\n")
#     f.close()
#     o.close()
#     l.close()
#
# convert('unzip/train-images-idx3-ubyte', 'unzip/train-labels-idx1-ubyte',
#         'mnist_train.csv', 60000)
# convert('unzip/t10k-images-idx3-ubyte', 'unzip/t10k-labels-idx1-ubyte',
#         'mnist_test.csv', 10000)
# ##########