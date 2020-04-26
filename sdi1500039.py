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
from sklearn.model_selection import train_test_split

# visualization
from wordcloud import WordCloud
from IPython.display import Image
from IPython.display import display

# for data exploration
import pandas as pd
import os
import numpy as np

# ## __Dataset Preprocessing__

# - ### *Make tsv files from all the txt files*

# region

pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 300)

myCategoriesFolder = ['business','entertainment','politics', 'sport', 'tech']
dataPathDir = './fulltext/data/'

myDataSetDf = pd.DataFrame(columns=['ID', 'TITLE',  'CONTENT',  'CATEGORY'])
id_count = 0

for category in myCategoriesFolder:
    # print('Folder ' + category + ':')
    specificPath = dataPathDir + category + '/'

    # list of dataframes of this category
    categoryFilesDfList = []

    # find the column's names of each csv
    for fileName in os.listdir(specificPath):
        # we need to check only .txt files
        if fileName.endswith(".txt"):
            # print(id_count, ' ', os.path.join(specificPath, fileName))
            id_count += 1
            # thisTxt = pd.read_csv(os.path.join(specificPath, fileName), dtype='unicode')
            thisTxt = open(os.path.join(specificPath, fileName),"r")
            thisTxtTitle = thisTxt.readline()
            # get rid of '\n' on the end of title line
            thisTxtTitle = thisTxtTitle.replace('\n', '')
            thisTxtContent = thisTxt.readlines()
            # https://stackoverflow.com/questions/1157106/remove-all-occurrences-of-a-value-from-a-list
            # get rid of empty lines '\n'
            thisTxtContent = list(filter(lambda a: a != '\n', thisTxtContent))
            # get rid of '\n' on the end of each line 
            thisTxtContent = [period.replace('\n', '') for period in thisTxtContent]
            # convert list of lines into a single string line
            thisTxtContent = ' '.join(thisTxtContent)
            myDataSetDf = myDataSetDf.append({'ID': id_count, 'TITLE': thisTxtTitle, 'CONTENT': thisTxtContent, 'CATEGORY': category.upper()}, ignore_index=True)
            thisTxt.close() 
display(myDataSetDf)
# endregion

# myDataSetDf = pd.read_csv("testData.tsv", sep='\t')

# ## __Make wordcloud for each category__

def makeWordCloud(myText, saveLocationPath, myMaxWords=100, myMask=None, myStopWords=None):
    '''Default function for generating wordcloud'''

    wc = WordCloud(background_color="white", mask=myMask, max_words=myMaxWords,
                   stopwords=myStopWords, contour_width=3, contour_color='steelblue')

    # generate word cloud
    wc.generate(myText)

    # store to file

    wc.to_file(saveLocationPath)

    return saveLocationPath


# - ### *Business Wordcloud*

# region

# to fill

# endregion

# - ### *Entertainment Wordcloud*

# region

# to fill

# endregion

# - ### *Politics Wordcloud*

# region

# to fill

# endregion

# - ### *Sport Wordcloud*

# region

# to fill

# endregion

# - ### *Tech Wordcloud*

# region

# to fill

# endregion

# ## __Classification__

# - ### *Split DataSet into TrainData and TestData*

# region

# trainDataSet, testDataSet = train_test_split(dataSetDf, test_size=0.2, stratify=dataSetDf['CATEGORY'])

# to be removed and use the above
trainDataSet, testDataSet = train_test_split(myDataSetDf, test_size=0.3, stratify=myDataSetDf['CATEGORY']) 
# to be removed and use the above

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

myDataSetDf

trainDataSet

testDataSet

testDataSetCategories
