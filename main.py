import songFeature
import classifiers
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import metrics
import pandas as pan


def getData(directory):
    data = pan.read_csv(directory)
    datas = data.drop('filename', axis=1)
    X_data = np.array(datas.drop(columns='label'))
    y_data = np.array(data['label'])
    n_class = datas.drop_duplicates(subset='label')['label']
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=True,stratify=y_data)
    return X_train, X_test, Y_train, Y_test, n_class

directory = "dataset/data.csv"

#load of dataset and selection of features
X_train, X_test, Y_train, Y_test, n_class = getData(directory)


#KNN
classif = classifiers.knnClassifier(X_train, Y_train)
pred = classif.predict(X_test)
print(classif)
metrics.val(Y_test, pred)
metrics.confMat(Y_test, pred, n_class,name='Confusion Matrix KNN')


#naive bayes
classif1 = classifiers.bayesianClassifier(X_train, Y_train)
pred = classif1.predict(X_test)
print(classif1)
metrics.val(Y_test, pred)
metrics.confMat(Y_test, pred, n_class, name='Confusion Matrix BC')



# extra tree classifier
classif2 = classifiers.extraTreesClassifier(X_train, Y_train)
pred = classif2.predict(X_test)
print(classif2)
metrics.val(Y_test, pred)
print(classification_report(Y_test, pred, target_names=n_class))
metrics.confMat(Y_test, pred, n_class, name='Confusion Matrix ETC')



# random forest classifier
classif3 = classifiers.randomForestClassifier(X_train, Y_train)
pred = classif3.predict(X_test)

metrics.val(Y_test, pred)
metrics.confMat(Y_test, pred, n_class, name='Confusion Matrix RFC')


calibers = classif3.feature_importances_ # weight of the features


std = np.std([tree.feature_importances_ for tree in classif3.estimators_], axis=0)
indexes = np.argsort(calibers)[::-1]


# build plot
std = np.std([tree.feature_importances_ for tree in classif3.estimators_],
             axis=0)
indexes = np.argsort(calibers)[::-1]

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indexes[f], calibers[indexes[f]]))
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), calibers[indexes],
       color="r", yerr=std[indexes], align="center")
plt.xticks(range(X_train.shape[1]), indexes)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# Prediction on a user's song 
a = 'yes'
while a == 'yes':
    print("\n")

    path_song = input("Insert the path of the desired song: ")
    print("LOADING...")
    song = songFeature.load_song(path_song)
    feature = songFeature.get_song_feature(song)
    print("Features processed:\n", feature, "\n\n")

    feature = feature.reshape(1, -1)
    print("Predictions:")
    print("\n")
    print("KNN: ", classifiers.getPred(classif, feature))
    print("Bayes: ", classifiers.getPred(classif1, feature))
    print("Extra Tree: ", classifiers.getPred(classif2, feature))
    print("Random Forest: ", classifiers.getPred(classif3, feature))
    a = input("Proceed with another song? [yes/no]  ")
