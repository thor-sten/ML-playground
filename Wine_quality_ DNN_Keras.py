# Wine quality data from http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
# https://www.datacamp.com/community/tutorials/deep-learning-python?utm_source=adwords_ppc&utm_campaignid=898687156&utm_adgroupid=48947256715&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=1t1&utm_creative=229765585186&utm_targetid=aud-299261629574:dsa-473406573755&utm_loc_interest_ms=&utm_loc_physical_ms=9062457&gclid=CjwKCAjw-8nbBRBnEiwAqWt1zSrmP0oRxUZaLkbOHiPoT9Oxukx9OX3TYGx9Y_CWt1ybO8zdnX72bBoC6CQQAvD_BwE

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

from keras.models import Sequential
from keras.layers import Dense



# Read in data 
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')


data_analysis = False

if data_analysis == True:
    # Print info
    print(white.info())
    print(red.info())

    # Overview of csv file:
    red.head()  # Show first rows
    red.tail()  # Show last rows
    red.sample(5)  # Take a sample of 5 rows
    red.describe()  # Statistics summary
    pd.isnull(red)  # Double check for null values


    # Visualizing the data (histogram), matplotlib:
    fig, ax = plt.subplots(1, 2)
    ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
    ax[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")

    # fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
    ax[0].set_ylim([0, 1000])
    ax[0].set_xlabel("Alcohol in % Vol")
    ax[0].set_ylabel("Frequency")
    ax[1].set_xlabel("Alcohol in % Vol")
    ax[1].set_ylabel("Frequency")
    #ax[0].legend(loc='best')
    #ax[1].legend(loc='best')
    fig.suptitle("Distribution of Alcohol in % Vol")

    plt.show()

    # Numpy:
    print(np.histogram(red.alcohol, bins=[7,8,9,10,11,12,13,14,15]))
    print(np.histogram(white.alcohol, bins=[7,8,9,10,11,12,13,14,15]))


    # Investigate wine quality vs. amount of sulfates (scatter plot)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].scatter(red['quality'], red["sulphates"], color="red")
    ax[1].scatter(white['quality'], white['sulphates'], color="white", edgecolors="black", lw=0.5)

    ax[0].set_title("Red Wine")
    ax[1].set_title("White Wine")
    ax[0].set_xlabel("Quality")
    ax[1].set_xlabel("Quality")
    ax[0].set_ylabel("Sulphates")
    ax[1].set_ylabel("Sulphates")
    ax[0].set_xlim([0,10])
    ax[1].set_xlim([0,10])
    ax[0].set_ylim([0,2.5])
    ax[1].set_ylim([0,2.5])
    fig.subplots_adjust(wspace=0.5)
    fig.suptitle("Wine Quality by Amount of Sulphates")

    plt.show()


    # Acidity
    np.random.seed(570)

    redlabels = np.unique(red['quality'])
    whitelabels = np.unique(white['quality'])

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    redcolors = np.random.rand(6,4)  # Label colors randomly chosen
    whitecolors = np.append(redcolors, np.random.rand(1,4), axis=0)

    for i in range(len(redcolors)):
        redy = red['alcohol'][red.quality == redlabels[i]]
        redx = red['volatile acidity'][red.quality == redlabels[i]]
        ax[0].scatter(redx, redy, c=redcolors[i])
    for i in range(len(whitecolors)):
        whitey = white['alcohol'][white.quality == whitelabels[i]]
        whitex = white['volatile acidity'][white.quality == whitelabels[i]]
        ax[1].scatter(whitex, whitey, c=whitecolors[i])
        
    ax[0].set_title("Red Wine")
    ax[1].set_title("White Wine")
    ax[0].set_xlim([0,1.7])
    ax[1].set_xlim([0,1.7])
    ax[0].set_ylim([5,15.5])
    ax[1].set_ylim([5,15.5])
    ax[0].set_xlabel("Volatile Acidity")
    ax[0].set_ylabel("Alcohol")
    ax[1].set_xlabel("Volatile Acidity")
    ax[1].set_ylabel("Alcohol") 
    #ax[0].legend(redlabels, loc='best', bbox_to_anchor=(1.3, 1))
    ax[1].legend(whitelabels, loc='best', bbox_to_anchor=(1.3, 1))
    #fig.suptitle("Alcohol - Volatile Acidity")
    fig.subplots_adjust(top=0.85, wspace=0.7)

    plt.show()


# Preprocess data
red['type'] = 1
white['type'] = 0
wines = red.append(white, ignore_index=True)  # ignore_index to continue labels without dublicates


if data_analysis == True:
    # Correlation matrix (seaborn)
    corr = wines.corr()
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
    sns.plt.show()


# Split the data up in train and test sets and standardize
X = wines.ix[:,0:11]  # Specify the data 
y = np.ravel(wines.type)  # Specify the target labels and flatten the array 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Set up model
model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(11,)))  # Input layer, dense = fully connected layer, Relu activation function (alternative tanh)
model.add(Dense(8, activation='relu'))  # One hidden layer 
model.add(Dense(1, activation='sigmoid'))  # Output layer


# Get model info
model.output_shape
model.summary()
model.get_config() 
model.get_weights()


# Compile and fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Binary cross. loss function. Metrics parameter to monitor the accuracy
model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)  # Train the model for 20 epochs, verbose = see progress bar logging


# Predict values
y_pred = model.predict(X_test)
y_pred[:5]
y_test[:5]  # Compare with test set


# Evaluate model
score = model.evaluate(X_test, y_test,verbose=1)
print(score)  # show loss and accuracy

confusion_matrix(y_test, y_pred)  # Shows correct and incorrect predictions
precision_score(y_test, y_pred)  # Measure of classifier’s exactness
recall_score(y_test, y_pred)  # Measure of classifier’s completeness
f1_score(y_test,y_pred)  # Weighted average of precision and recall.
cohen_kappa_score(y_test, y_pred)  # Classification accuracy normalized by the imbalance of the classes in the data

