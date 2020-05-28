# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
def word_features(word):
    return {'items': word}
from nltk import MaxentClassifier
numIterations = 100
algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
# needs data set in form of list(tuple(dict, str))
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #review = ' '.join(review)
    if(dataset['Liked'][i]==0):
        result = "negative";
    else:
        result = "positive" 
    for word in review:
        corpus.append((word_features(word), result))

# Creating the Bag of Words model
#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(max_features = 1500)
#X = cv.fit_transform(corpus).toarray()
#y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(corpus, test_size = 0.20, random_state = 0)

# Fitting maxentropy to the Training set
classifier = nltk.MaxentClassifier.train(X_train, algorithm, max_iter=numIterations)

# Predicting the Test set results
y_pred = classifier.classify(word_features("first"))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(nltk.classify.accuracy(classifier, X_test))

#Maximum Entropy

#Accuracy = 0.73

#Precision = 0.684

#Recall = 0.883

#F1 Score = 0.771