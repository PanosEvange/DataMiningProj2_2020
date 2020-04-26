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
# endregion

# ## __Dataset Preprocessing__

# - ### *Make tsv files from all the txt files*

# region

# to fill

# endregion

dataSetDf = pd.read_csv("testData.tsv", sep='\t')

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
trainDataSet, testDataSet = train_test_split(dataSetDf, test_size=0.3, stratify=dataSetDf['CATEGORY']) 
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

dataSetDf

trainDataSet

testDataSet

testDataSetCategories
