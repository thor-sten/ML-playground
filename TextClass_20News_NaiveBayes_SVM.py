# https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
# https://towardsdatascience.com/a-production-ready-multi-class-text-classifier-96490408757

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.stem.snowball import SnowballStemmer


#Loading the data set - training data
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

# Visualize data and categories
print('Categories:', twenty_train.target_names)
print("\n".join(twenty_train.data[0].split("\n")[:3]))


# Preprocessing

# Bag-of-words: Convert text documents to a matrix of token counts
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print('BOW matrix dimension:', X_train_counts.shape)

# TF-IDF (Term Frequency - times inverse document frequency)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


# Classification

# Naive Bayes (NB) classifier
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

# Do all of the above with less code by building a pipeline:
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(twenty_test.data)
print('Naive Bayes accuracy:', np.mean(predicted == twenty_test.target)) # --> 77.4%


# Support Vector Machines - http://scikit-learn.org/stable/modules/svm.html
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])
text_clf_svm = text_clf_svm.fit(twenty_train.data, twenty_train.target)
predicted_svm = text_clf_svm.predict(twenty_test.data)
print('Support Vector Machines accuracy:', np.mean(predicted_svm == twenty_test.target))  # --> 82.4%


# Optimization

# Grid Search (hyperparameter optimization) for NB
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)

print('Best mean score (NB):', gs_clf.best_score_)  # increased to 90.6%
print('Best parameters (NB):', gs_clf.best_params_)

# Grid search for SVM
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}
gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(twenty_train.data, twenty_train.target)

print('Best mean score (SVM):', gs_clf_svm.best_score_)  # increased to 89.8%
print('Best parameters (SVM):', gs_clf_svm.best_params_)


# Further optimization methods

# NLTK, removing stop words (the, then etc) from the data
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
# --> increase from 77.4% to 81.7%

# Stemming words with Snoball stemmer
nltk.download('stopwords')
stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
    
stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

# Use uniform prior for NB
text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()), ('mnb', MultinomialNB(fit_prior=False))])
text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)
predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)

print('Accuracy after stemming:', np.mean(predicted_mnb_stemmed == twenty_test.target))  # --> 81.7%

# Dimensionality reduction with Singular Value Decomposition to find most important words?

# Write data to csv file, together with classification
newdata = []
for i,row in enumerate(twenty_train.data):
    newdata.append([row, data.target[i]])

with open('20Newsgroup.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerows(newdata)

