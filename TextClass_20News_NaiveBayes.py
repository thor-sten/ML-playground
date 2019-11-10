import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load data: Text in column 1, classifier in column 2
data = pd.read_csv('20Newsgroup.csv', delimiter=';')
numpy_array = data.as_matrix()
X = numpy_array[:,0]
Y = numpy_array[:,1]

# Convert labels to string for model fitting
Y = list(map(str, Y))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

# Naive Bayes
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(X_train,Y_train)

predicted = text_clf.predict(X_test)
print('Accuracy:', np.mean(predicted == Y_test))