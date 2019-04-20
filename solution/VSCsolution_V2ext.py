#import pickle file of labelled data 
import nltk, numpy
import sklearn
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pickle
import textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import sys
import pandas.core.indexes

sys.modules['pandas.indexes'] = pandas.core.indexes 
pickle_in2 = open("/Users/felineweise/Desktop/Programming/CodingChallenge/datascience-challenge-master/data/labelled_dataset.pickle", 'rb')
data2 = pickle.load(pickle_in2)
data2.shape
data2.head()

# create a dataframe using texts and lables
DF2 = pandas.DataFrame(data2)
DF2['text'] = data2.text
DF2['label'] = data2.labelmax

DF2.head()

DF2 = DF2.drop(["labelmax"], axis=1)
DF2.head()
DF2.shape

#remove all 'null' values from dataframe to avoid errors during labelencoding
import numpy as np
from numpy import nan
DF2['label'] = DF2['label'].replace(to_replace = "null", value= np.nan)
nulls = DF2[DF2['label'].isna()].index.tolist()
DF2 = DF2.dropna(how='any',axis=0) 
DF2.shape

#import csv file of unlabelled data (extracted from separate json files in R)
import pandas 

data_unlabelled = pandas.read_csv("/Users/felineweise/Desktop/Programming/CodingChallenge/datascience-challenge-master/data_unlabelled", encoding='latin-1')
data_unlabelled

DF_unlab = pandas.DataFrame(data_unlabelled)
DF_unlab.head()
list(DF_unlab.columns.values)
DF_unlab['ID'] = DF_unlab['Unnamed: 0']
DF_unlab['text'] = DF_unlab['x']
DF_unlab = DF_unlab.drop(['Unnamed: 0', 'x'], axis=1)
DF_unlab.head()

import nltk
import re

#prepare function to clean the texts 
wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english', 'german')

def normalize_reviews(doc): 
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[-]', ' ', doc)
    doc = re.sub('Cons', '', doc)
    doc = re.sub('Pros', '', doc)
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc


normalize_text = np.vectorize(normalize_reviews)

#texts of labelled dataset is cleaned
DF2['text'] = normalize_text(DF2['text'])

#these are the cleaned unlabelled texts in a numpy array
DF_unlab['text'] = normalize_text(DF_unlab['text']) 
type(DF_unlab)
DF_unlab.head
DF_unlab.shape


#prepare data for feature engineering, train/test runs and different models
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn import linear_model
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

##extract TF-IDF Vectors as features in terms of scores that represent the relative importance of every word in different documents  
tfidf_vect = TfidfVectorizer(analyzer='word' ,encoding = 'latin-1', ngram_range= (1,3), max_features= 35000)

#fit text data in labelled dataset to the TF-IDF vectors as features  
tfidf_vect.fit(DF2['text'])

# split the labelled dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = sklearn.model_selection.train_test_split(DF2['text'], DF2['label'])

#transform train/test splits to contain word level features 
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

#also apply this on text data in unlabelled dataset 
tfidf_vect.fit(DF_unlab['text'])
unlabelled_tfidf = tfidf_vect.transform(DF_unlab['text'])


# label encode  target variable 
encoder = preprocessing.LabelEncoder()
encoder.fit(['adaptability', 'collaboration', 'customer', 'detail', 'integrity', 'result'])
list(encoder.classes_)

train_y = encoder.fit_transform(train_y) 
valid_y = encoder.fit_transform(valid_y)

#to check which label corresponds to which dimensions 
DF2['label_id'] = encoder.fit_transform(DF2['label'])

DFlabel_id = DF2[['label', 'label_id']].drop_duplicates().sort_values('label_id')
label_to_id = dict(DFlabel_id.values)
# mapping is as follows {'adaptability': 0, 'collaboration': 1, 'customer': 2, 'detail': 3, 'integrity': 4, 'result': 5}

#use sklearn.feature_selection.chi2 to find the features that are the most correlated with each dimension/label
from sklearn.feature_selection import chi2
import numpy as np

##brief excursus: extract TF-IDF Word-level features in labelled dataset and check which words correlate most with which dimension
DF2_tfidf = TfidfVectorizer(min_df=5, encoding='latin-1', ngram_range=(1, 3), analyzer = 'word',  max_features=7500)
features = DF2_tfidf.fit_transform(DF2.text).toarray()
labels = DF2.label_id
features.shape
import sys

N = 2
labels = DF2['label']
for label, label_id in sorted(label_to_id.items()):
  features_chi2 = chi2(features, labels == label_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(DF2_tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  trigrams = [v for v in feature_names if len(v.split(' ')) == 3]
  sys.stdout=open("Correlates dimensions - word-level features.txt","a")
  print("# '{}':".format(label))
  sys.stdout.close() 
  sys.stdout=open("Correlates dimensions - word-level features.txt","a")
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  sys.stdout.close() 
  sys.stdout=open("Correlates dimensions - word-level features.txt","a")
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
  sys.stdout.close() 
  sys.stdout=open("Correlates dimensions - word-level features.txt","a")
  print("  . Most correlated trigrams:\n. {}".format('\n. '.join(trigrams[-N:])))   
sys.stdout.close() 
#the results of this are saved in a text file and included in the "solution" folder I compiled 


#1. fit DIFFERENT models onto train/valid split of labelled data and note accuracy for predictions on validate split of labelled data 
#2. afterwards, fit model onto all text data of labelled data and make predictions on unlabelled data 
#3. save predictions made into file 

#1.1 fit NAIVE BAYES model
NaiveBayes = MultinomialNB().fit(xtrain_tfidf, train_y)

#let NAIVE BAYES model perform predictions on validation split of labelled data 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

NBpredicted_label = NaiveBayes.predict(xvalid_tfidf)

print('accuracy %s' % accuracy_score(NBpredicted_label, valid_y)) #accuracy 0.5638344609436781

#1.2 fit RANDOM FOREST model 
from sklearn.ensemble import RandomForestClassifier

RndmForest = RandomForestClassifier().fit(xtrain_tfidf, train_y)

#let Random forest model perform predictions on validation split of labelled data 

RndmForestpredicted_label = RndmForest.predict(xvalid_tfidf)

print('accuracy %s' % accuracy_score(RndmForestpredicted_label, valid_y)) #accuracy 0.5027238379755733

#1.3 fit LOGISTIC REGRESSION model 
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.linear_model import LogisticRegression

LogReg = LogisticRegression().fit(xtrain_tfidf, train_y)

#let LOGISTIC REGRESSION model perform predictions on validation split of labelled data 

LogRegpredicted_label = LogReg.predict(xvalid_tfidf)

print('accuracy %s' % accuracy_score(LogRegpredicted_label, valid_y)) #accuracy 0.7723398646867586


#1.4 fit Linear SUPPORT VECTOR CLASSIFICATION (SVC) model 
from sklearn.svm import LinearSVC

LinSVC = LinearSVC().fit(xtrain_tfidf, train_y)

#let Linear (SVC) model perform predictions on validation split of labelled data 

LinSVCpredicted_label = LinSVC.predict(xvalid_tfidf)

print('accuracy %s' % accuracy_score(LinSVCpredicted_label, valid_y)) #accuracy 0.7790176610139706



#2. SINCE Linear SUPPORT VECTOR CLASSIFICATION SEEMS TO YIELD HIGHEST ACCURACY (closely followed by Logistic Regression), I WILL PROCEDE WITH THIS MODEL IN ORDER TO CLASSIFY UNSEEN AND UNLABELLED DATA 

#unlabelled_tfidf is 'xvalid' 
unlabelled_tfidf

LinSVCpredicted_unlabelled = LogReg.predict(unlabelled_tfidf)
print(LinSVCpredicted_unlabelled)
type(LinSVCpredicted_unlabelled)
LinSVCpredicted_unlabelled.shape

array_unlabelledpredicted = numpy.transpose(LinSVCpredicted_unlabelled)


out = encoder.inverse_transform(LinSVCpredicted_unlabelled)

#create a new numpy array with unlabelled texts and predicted dimensions as columns 
unlabelled_texts = np.array(DF_unlab['text'])
results = np.column_stack((unlabelled_texts, out))

results = pandas.DataFrame(results, columns = ["reviews","predicted dimensions" ] )
list(results.columns.values)
results.head()

import pandas as pd 
pd.DataFrame(results).to_csv("/Users/felineweise/Desktop/Programming/CodingChallenge/datascience-challenge-master/Predicted dimensions of unlabelled text data.csv")


##How could the classifier be improved?? 

#ONE COULD IMPROVE THE CLASSIFIER BY INCLUDING MORE FEATURES SUCH AS WORD EMBEDDINGS OR NATURAL LANGUAGE BASED FEATURES 
#BY PERFORMING DIMENSIONALITY REDUCTION (FOR EXAMPLE BY USING PRINCIPAL COMPONENT ANALYSIS OR THE LIKE) ON ADDITIONAL FEATURES 
##ONE COULD THEN INCLUDE ONLY THOSE FEATURES THAT ARE MOST DISCRIMINATIVE FOR THE 6 DIMENSIONS OF COMPANY CULTURE

##How can the classifier be made available in production?

#IN ORDER TO MAKETHE CLASSIFIER AVAILABLE FOR PRODUCTION AN APPLICATION PROGRAMMING INTERFACE (API) WOULD HAVE TO BE BUILT
#THE API WOULD HAVE TO HAVE AN ENDPOINT TO GLASSDOOR SO THAT THE WEBSITE CAN POST THEIR REVIEWS TO THE ENDPOINT
#THE MODEL THAT IS COPIED TO THE ROOT OF THE API COULD THEN BE USED TO IDENTIFY WHICH DIMENSION OF COMPANY CULTURE THE REVIEW IS RELATED TO 