#!/usr/bin/python

""" lecture and example code for decision tree unit """

import sys
sys.path.append("../../choose_your_own/")

from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifyDT import classify

features_train, labels_train, features_test, labels_test = makeTerrainData()


# the classify() function in classifyDT is where the magic
# happens--fill in this function in the file 'classifyDT.py'!
clf = classify(features_train, labels_train)

# from sklearn import tree
# clf2 = tree.DecisionTreeClassifier(min_samples_split=2)
# clf50 = tree.DecisionTreeClassifier(min_samples_split=50)
# clf2 = clf2.fit(features_train, labels_train)
# clf50 = clf50.fit(features_train, labels_train)
# print "Accuracy clf2: ", clf2.score(features_test, labels_test)
# print "Accuracy clf50: ", clf50.score(features_test, labels_test)

# grader code, do not modify below this line
acc = clf.score(features_test, labels_test)


prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())

print "Accuracy: ", acc
