# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_markers: region,endregion
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # <center>Data Mining Project 2 Spring semester 2019-2020</center>
# ## <center>Παναγιώτης Ευαγγελίου &emsp; 1115201500039</center>
# ## <center>Γεώργιος Μαραγκοζάκης &emsp; 1115201500089</center>

# ___

# ### Do all the necessary imports for this notebook

# region
# data processing
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords as nltkStopwords
from string import punctuation, digits
import re
from nltk import word_tokenize
from nltk.stem import PorterStemmer

# visualization
from wordcloud import WordCloud
from IPython.display import Image
from IPython.display import display
from itertools import cycle

# classification
from sklearn.model_selection import KFold
from sklearn import svm, preprocessing
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import scipy
from collections import Counter

# vectorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# clustering
from nltk.cluster import KMeansClusterer, cosine_distance
from sklearn.decomposition import PCA

# for data exploration
import os
import numpy as np
# endregion

# ## __Dataset Preprocessing__

# - ### *Make tsv files from all the txt files*

# region
myCategoriesFolder = ['business','entertainment','politics', 'sport', 'tech']
dataPathDir = './fulltext/data/'

myDataSetDf = pd.DataFrame(columns=['ID', 'TITLE',  'CONTENT',  'CATEGORY'])
id_count = 0

for category in myCategoriesFolder:
    specificPath = dataPathDir + category + '/'

    # find the column's names of each csv
    for fileName in os.listdir(specificPath):
        # we need to check only .txt files
        if fileName.endswith(".txt"):
            
            thisTxt = open(os.path.join(specificPath, fileName),"r")
            thisTxtTitle = thisTxt.readline()

            # get rid of '\n' on the end of title line
            thisTxtTitle = thisTxtTitle.replace('\n', '')

            thisTxtContent = thisTxt.readlines()

            # get rid of empty lines '\n'
            thisTxtContent = list(filter(lambda a: a != '\n', thisTxtContent))

            # get rid of '\n' on the end of each line 
            thisTxtContent = [period.replace('\n', '') for period in thisTxtContent]

            # convert list of lines into a single string line
            thisTxtContent = ' '.join(thisTxtContent)

            myDataSetDf = myDataSetDf.append({'ID': id_count, 'TITLE': thisTxtTitle, 'CONTENT': thisTxtContent, 'CATEGORY': category.upper()}, ignore_index=True)
            thisTxt.close()

            id_count += 1

display(myDataSetDf)
# endregion

# ## __Make wordcloud for each category__

def makeWordCloud(myText, saveLocationPath, myMaxWords=100, myMask=None, myStopWords=None):
    '''Default function for generating wordcloud'''

    wc = WordCloud(background_color="white", mask=myMask, max_words=myMaxWords,
                   stopwords=myStopWords, contour_width=3, contour_color='steelblue',
                   width=600, height=600)

    # generate word cloud
    wc.generate(myText)

    # store to file

    wc.to_file(saveLocationPath)

    return saveLocationPath

def columnToText(myDfColumn):
    wholeColumnText = ''

    for text in myDfColumn:
        wholeColumnText = wholeColumnText + ' ' + text

    return wholeColumnText

stopWords = ENGLISH_STOP_WORDS
myAdditionalStopWords = ['say','said', 'new', 'need', 'year']
stopWords = (stopWords.union(myAdditionalStopWords))

# - ### *Business Wordcloud*

# region
makeWordCloud(saveLocationPath="businessWordCloud.png", myText=columnToText(myDataSetDf[myDataSetDf['CATEGORY'] == "BUSINESS"]['CONTENT']), myStopWords=stopWords)

Image('businessWordCloud.png')
# endregion

# - ### *Entertainment Wordcloud*

# region
makeWordCloud(saveLocationPath="entertainmentWordCloud.png", myText=columnToText(myDataSetDf[myDataSetDf['CATEGORY'] == "ENTERTAINMENT"]['CONTENT']), myStopWords=stopWords)

Image('entertainmentWordCloud.png')
# endregion

# - ### *Politics Wordcloud*

# region
makeWordCloud(saveLocationPath="politicsWordCloud.png", myText=columnToText(myDataSetDf[myDataSetDf['CATEGORY'] == "POLITICS"]['CONTENT']), myStopWords=stopWords)

Image('politicsWordCloud.png')
# endregion

# - ### *Sport Wordcloud*

# region
makeWordCloud(saveLocationPath="sportWordCloud.png", myText=columnToText(myDataSetDf[myDataSetDf['CATEGORY'] == "SPORT"]['CONTENT']), myStopWords=stopWords)

Image('sportWordCloud.png')
# endregion

# - ### *Tech Wordcloud*

# region
makeWordCloud(saveLocationPath="techWordCloud.png", myText=columnToText(myDataSetDf[myDataSetDf['CATEGORY'] == "TECH"]['CONTENT']), myStopWords=stopWords)

Image('techWordCloud.png')
# endregion

# ## __Classification__

def scoresReportCv(clf, trainX, trainY):
    """
    Printing scores using cross_val_score    
    """

    print('----Report for 10-fold Cross Validation----')

    precisions = cross_val_score(clf, trainX, trainY, cv=10, scoring='precision_weighted')
    print ('Precision \t %0.2f' % (np.mean(precisions)))

    recalls = cross_val_score(clf, trainX, trainY, cv=10, scoring='recall_weighted')
    print ('Recalls \t %0.2f' % (np.mean(recalls)))

    f1s = cross_val_score(clf, trainX, trainY, cv=10, scoring='f1_weighted')
    print ('F-Measure \t %0.2f' % (np.mean(f1s)))

    scores = cross_val_score(clf, trainX, trainY, cv=10)
    print("Accuracy: \t %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def makeRocPlot(labelTest, predictions, labelEncoder):
    # Binarize the output
    labelsAsNumber = [i for i in range(0,len(labelEncoder.classes_))]
    labelTest = label_binarize(labelTest, classes=labelsAsNumber)
    n_classes = labelTest.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labelTest[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labelTest.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw = 2

    # Plot all ROC curves
    plt.figure(figsize=(12, 12))
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'forestgreen', 'maroon'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(labelEncoder.classes_[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC plot of all classes')
    plt.legend(loc="lower right")
    plt.show()

#   - #### Classification using SVM classifier

def SvmClassification(trainX, trainY, testX, testY, labelEncoder):
    """
    Classify the text using the SVM classifier of scikit-learn    
    """
    
    clf = svm.SVC(kernel='linear', C=1, probability=True)

    # fit train set
    clf.fit(trainX, trainY)
    
    # use 10-fold Cross Validation
    scoresReportCv(clf, trainX, trainY)

    # Predict test set
    predY = clf.predict(testX)

    # Classification_report
    print('\n----Report for predictions on test dataset----')
    print(classification_report(testY, predY, target_names=list(labelEncoder.classes_)))

    print('\n----ROC plot for predictions on test dataset----')
    y_score = clf.predict_proba(testX)

    makeRocPlot(testY, y_score, labelEncoder)

    return accuracy_score(testY, predY)

#   - #### Classification using Random Forests classifier

def RandomForestClassification(trainX, trainY, testX, testY, labelEncoder):
    """
    Classify the text using the Random Forest classifier of scikit-learn    
    """
    
    clf = RandomForestClassifier()
        
    # fit train set
    clf.fit(trainX, trainY)
    
    # use 10-fold Cross Validation
    scoresReportCv(clf, trainX, trainY)

    # Predict test set
    predY = clf.predict(testX)

    # Classification_report
    print('\n----Report for predictions on test dataset----')
    print(classification_report(testY, predY, target_names=list(labelEncoder.classes_)))

    print('\n----ROC plot for predictions on test dataset----')
    y_score = clf.predict_proba(testX)

    makeRocPlot(testY, y_score, labelEncoder)

    return accuracy_score(testY, predY)

#   - #### Classification using Naive Bayes classifier

def NaiveBayesClassification(trainX, trainY, testX, testY, labelEncoder):
    """
    Classify the text using the Naive Bayes classifier of scikit-learn    
    """

    clf = GaussianNB()
    
    trainX = trainX.toarray()
    
    # fit train set
    clf.fit(trainX, trainY)
    
    # use 10-fold Cross Validation
    scoresReportCv(clf, trainX, trainY)

    # Predict test set
    testX = testX.toarray()
    predY = clf.predict(testX)

    # Classification_report
    print('\n----Report for predictions on test dataset----')
    print(classification_report(testY, predY, target_names=list(labelEncoder.classes_)))

    print('\n----ROC plot for predictions on test dataset----')

    y_score = clf.predict_proba(testX)

    makeRocPlot(testY, y_score, labelEncoder)

    return accuracy_score(testY, predY)

#   - #### Classification using K-Nearest Neighbor classifier

# Our implemantion is based on this link https://towardsdatascience.com/k-nearest-neighbor-classifier-from-scratch-in-python-698e3de97063 
# and this link https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

# region
# calculate the Euclidean distance between two 1d-arrays
def distance(instance1, instance2):
    return scipy.spatial.distance.euclidean(instance1, instance2)

def get_neighbors(training_set, 
                  labels, 
                  test_instance, 
                  k, 
                  distance=distance):
    """
    get_neighors calculates a list of the k nearest neighbors
    of an instance 'test_instance'.
    The list neighbors contains 3-tuples with  
    (index, dist, label)
    where 
    index    is the index from the training_set, 
    dist     is the distance between the test_instance and the 
             instance training_set[index]
    distance is a reference to a function used to calculate the 
             distances
    """
    distances = []
    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append((training_set[index], dist, labels[index]))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors

# The function 'vote' returns the most common class. (Majority Voting)
def vote(neighbors):
    class_counter = Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]] += 1
    return class_counter.most_common(1)[0][0]

# Make a prediction with neighbors
def predict_classification(training_set, labels, test_instance, k, distance=distance):
    neighbors = get_neighbors(training_set, labels, test_instance, k, distance=distance)
    prediction = vote(neighbors)
    return prediction

# kNN Algorithm
def k_nearest_neighbors(trainX, trainY, testX, num_neighbors):
    predictions = list()
    for row in testX:
        output = predict_classification(trainX, trainY, row,  num_neighbors, distance=distance )
        predictions.append(output)
    return(predictions)

def KnnClassification(trainX, trainY, testX, testY, labelEncoder):
    """
    Classify the text using the KNN classifier we implemented   
    """

    trainXarray = trainX.toarray()
    testXarray = testX.toarray()

    predictions = k_nearest_neighbors(trainXarray, trainY, testXarray, 100)

    predY = np.asarray(predictions)

    # Classification_report
    print('\n----Report for predictions on test dataset----')
    print(classification_report(testY, predY, target_names=list(labelEncoder.classes_)))

    return accuracy_score(testY, predY)

# endregion
# - ### *Split DataSet into TrainData and TestData*

# region
trainDataSet, testDataSet = train_test_split(myDataSetDf, test_size=0.2, stratify=myDataSetDf['CATEGORY'])

# reset index
trainDataSet.reset_index(drop=True, inplace=True)
testDataSet.reset_index(drop=True, inplace=True)

# save to tsv files
trainDataSet.to_csv('train_set.tsv', sep = '\t')

# save test_set categories
testDataSetCategories = testDataSet[['CATEGORY']].copy()

testDataSetCategories.to_csv('test_set_categories.tsv', sep = '\t')

testDataSet = testDataSet.drop('CATEGORY', axis=1)
testDataSet.to_csv('test_set.tsv', sep = '\t')
# endregion

# Prepare train and test data that we will need below

# region
# build label encoder for categories
le = preprocessing.LabelEncoder()
le.fit(trainDataSet["CATEGORY"])

# transform categories into numbers
trainY = le.transform(trainDataSet["CATEGORY"])
testY = le.transform(testDataSetCategories["CATEGORY"])

accuracyDict = dict()
# endregion

# ## __Vectorization__

# Let's do classification using 2 different ways of vectorization

# region language="javascript"
# IPython.OutputArea.prototype._should_scroll = function(lines) {
#     return false;
# }
# endregion

#   - #### Bag-of-words vectorization

# region
bowVectorizer = CountVectorizer(max_features=1000)

trainX = bowVectorizer.fit_transform(trainDataSet['CONTENT'])
testX = bowVectorizer.transform(testDataSet['CONTENT'])

print('-------------SVM Classification with BOW Vectorization-------------')
accuracyDict["BOW-SVM"] = SvmClassification(trainX, trainY, testX, testY, le)

print('\n-------------Random Forests Classification with BOW Vectorization-------------')
accuracyDict["BOW-RandomForests"] = RandomForestClassification(trainX, trainY, testX, testY, le)

print('\n-------------Naive Bayes Classification with BOW Vectorization-------------')
accuracyDict["BOW-NB"] = NaiveBayesClassification(trainX, trainY, testX, testY, le)

print('\n-------------K Nearest Neighbor Classification with BOW Vectorization-------------')
accuracyDict["BOW-knn"] = KnnClassification(trainX, trainY, testX, testY, le)
# endregion

#   - #### Tf-idf vectorization

# region
tfIdfVectorizer = TfidfVectorizer(max_features=1000)

trainX = tfIdfVectorizer.fit_transform(trainDataSet['CONTENT'])
testX = tfIdfVectorizer.transform(testDataSet['CONTENT'])

print('-------------SVM Classification with TfIdf Vectorization-------------')
accuracyDict["TfIdf-SVM"] = SvmClassification(trainX, trainY, testX, testY, le)

print('\n-------------Random Forests Classification with TfIdf Vectorization-------------')
accuracyDict["TfIdf-RandomForests"] = RandomForestClassification(trainX, trainY, testX, testY, le)

print('\n-------------Naive Bayes Classification with TfIdf Vectorization-------------')
accuracyDict["TfIdf-NB"] = NaiveBayesClassification(trainX, trainY, testX, testY, le)

print('\n-------------K Nearest Neighbor Classification with TfIdf Vectorization-------------')
accuracyDict["TfIdf-knn"] = KnnClassification(trainX, trainY, testX, testY, le)
# endregion

#   #### Results Summary

# region
resultsData = {r'Vectorizer \ Classifier': ['BOW', 'Tfidf'],
               'SVM': [accuracyDict["BOW-SVM"], accuracyDict["TfIdf-SVM"]],
               'Random Forest': [accuracyDict["BOW-RandomForests"], accuracyDict["TfIdf-RandomForests"]],
               'Naive Bayes': [accuracyDict["BOW-NB"], accuracyDict["TfIdf-NB"]],
               'K Nearest Neighbor': [accuracyDict["BOW-knn"], accuracyDict["TfIdf-knn"]]}

resultsDataFrame = pd.DataFrame(data=resultsData)

resultsDataFrame
# endregion

# ## __Beat the Benchmark (bonus)__

# region
def preprocessText(initText):
    """Preprocess the text"""

    # Make everything to lower case
    processedText = initText.lower()

    # Remove urls
    processedText = re.sub(r'(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)'
                           r'*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?', ' ', processedText)

    # Remove any punctuation from the text
    for c in punctuation:
        processedText = processedText.replace(c, ' ')

    # Remove digits
    processedText = re.sub(r'\d+', '', processedText)

    # Remove consecutive spaces
    processedText = re.sub(r" {2,}", ' ', processedText)    

    # Split to words
    tokens = word_tokenize(processedText)

    # Remove sropwords
    stopWords = ENGLISH_STOP_WORDS
    stopWords = (stopWords.union(nltkStopwords.words('english')))
    filtered = [w for w in tokens if w not in stopWords]

    # Concat the remaining words in a single string again
    if not filtered:  # list is empty
        processedText = ''
    else:
        processedText = filtered[0]
        for word in filtered[1:]:
            processedText = processedText + ' ' + word

    return processedText

def stemmingPreprocess(initText):
    # Split to words
    tokens = word_tokenize(initText)

    # Do the stemming
    stemmer = PorterStemmer()
    stems = [stemmer.stem(token) for token in tokens]

    # Concat the remaining words in a single string again
    if not stems:  # list is empty
        processedText = ''
    else:
        processedText = stems[0]
        for stem in stems[1:]:
            processedText = processedText + ' ' + stem

    return processedText
# endregion

# Let's do some preprocessing for train and test data

# region
trainDataPreprocessed = trainDataSet.copy()
testDataPreprocessed = testDataSet.copy()

# preprocess train data
for index, row in trainDataPreprocessed.iterrows():
    initialText = row["CONTENT"]
    trainDataPreprocessed.iloc[index]["CONTENT"] = preprocessText(initialText)

# preprocess test data
for index, row in testDataPreprocessed.iterrows():
    initialText = row["CONTENT"]
    testDataPreprocessed.iloc[index]["CONTENT"] = preprocessText(initialText)
# endregion

# Let's do stemming

# region
for index, row in trainDataPreprocessed.iterrows():
    initialText = row["CONTENT"]
    trainDataPreprocessed.iloc[index]["CONTENT"] = stemmingPreprocess(initialText)

for index, row in testDataPreprocessed.iterrows():
    initialText = row["CONTENT"]
    testDataPreprocessed.iloc[index]["CONTENT"] = stemmingPreprocess(initialText)
# endregion

# We will check only the SVM classifier with Tf-idf vectorization

# region
tfIdfVectorizer = TfidfVectorizer(max_features=1000)

trainX = tfIdfVectorizer.fit_transform(trainDataPreprocessed['CONTENT'])
testX = tfIdfVectorizer.transform(testDataPreprocessed['CONTENT'])

print('\n-------------SVM Classification with TfIdf Vectorization in processed text-------------')
accuracyDict["TfIdf-SVM-processed"] = SvmClassification(trainX, trainY, testX, testY, le)
# endregion

# Let's compare scores

# region
resultsDataCompare = {'SVM without preprocessing': [accuracyDict["TfIdf-SVM"]],
               'SVM with preprocessing': [accuracyDict["TfIdf-SVM-processed"]]}

resultsCompareDataFrame = pd.DataFrame(data=resultsDataCompare)

resultsCompareDataFrame
# endregion

# As we see there is no big difference between scores for max_features=1000 in TfidfVectorizer. 
# Let's check what happens for max_features=100

tfIdfVectorizer = TfidfVectorizer(max_features=100)

# region
trainX = tfIdfVectorizer.fit_transform(trainDataSet['CONTENT'])
testX = tfIdfVectorizer.transform(testDataSet['CONTENT'])

print('-------------SVM Classification with TfIdf Vectorization for max_features=100-------------')
accuracyDict["TfIdf-SVM-100"] = SvmClassification(trainX, trainY, testX, testY, le)
# endregion

# region
trainX = tfIdfVectorizer.fit_transform(trainDataPreprocessed['CONTENT'])
testX = tfIdfVectorizer.transform(testDataPreprocessed['CONTENT'])

print('\n-------------SVM Classification with TfIdf Vectorization in processed text for max_features=100-------------')
accuracyDict["TfIdf-SVM-processed-100"] = SvmClassification(trainX, trainY, testX, testY, le)
# endregion

# Let's compare scores one more time

# region
resultsDataCompare = {'SVM without preprocessing for max_features=100': [accuracyDict["TfIdf-SVM-100"]],
                      'SVM with preprocessing for max_features=100': [accuracyDict["TfIdf-SVM-processed-100"]]}

resultsCompareDataFrame = pd.DataFrame(data=resultsDataCompare)

resultsCompareDataFrame
# endregion

# Here we can see a significant difference.

# ## __Clustering__

# region

# demo from nltk.org about understanding kmeans with euklidean
# https://www.nltk.org/_modules/nltk/cluster/kmeans.html#demo

def KmeansClustering(trainX, numberOfClusters, numberOfRepeats):
    # init cluster with trainX
    # maybe init with something smarter than none ? https://datascience.stackexchange.com/questions/5656/k-means-what-are-some-good-ways-to-choose-an-efficient-set-of-initial-centroids
    clusterer = KMeansClusterer(numberOfClusters, cosine_distance, initial_means=None, repeats=numberOfRepeats)
    assigned_clusters = clusterer.cluster(trainX, assign_clusters=True)
    return clusterer, assigned_clusters
# endregion

# ### PCA

def principalComponentAnalysis(nComponents, trainX, labels, clusters):
    # reduce the features to 2D
    random_state = 0 
    pca = PCA(n_components=nComponents, random_state=random_state)
    reduced_features = pca.fit_transform(trainX.toarray())

    # reduce the cluster centers to 2D
    reduced_cluster_centers = pca.transform(labels._means)

    plt.figure(figsize=(12, 12))
    plt.scatter(reduced_features[:,0], reduced_features[:,1], c=clusters)
    plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')

# ### BoW

# region

bowVectorizer = CountVectorizer(max_features=1000)

trainX = bowVectorizer.fit_transform(trainDataSet['CONTENT'])
testX = bowVectorizer.transform(testDataSet['CONTENT'])

# convert trainX, testX into list of arrays
vectorsTrainX = [np.array(f) for f in trainX.toarray()]
vectorsTestX = [np.array(f) for f in testX.toarray()]
labels, clusters = KmeansClustering(vectorsTrainX, 5, 20)
principalComponentAnalysis(2, trainX, labels, clusters)

# endregion

# ### tf-idf

# region

tfIdfVectorizer = TfidfVectorizer(max_features=1000)

trainX = tfIdfVectorizer.fit_transform(trainDataSet['CONTENT'])
testX = tfIdfVectorizer.transform(testDataSet['CONTENT'])

# convert trainX, testX into list of arrays
vectorsTrainX = [np.array(f) for f in trainX.toarray()]
vectorsTestX = [np.array(f) for f in testX.toarray()]
labels, clusters = KmeansClustering(vectorsTrainX, 5, 40)
principalComponentAnalysis(2, trainX, labels, clusters)

# endregion
