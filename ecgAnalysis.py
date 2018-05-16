import numpy as np
import csv as csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing as prep
from sklearn import svm

# Load the dataset with given file name, and split it into train and test set
def loadDataset(fileName, split=False, returnIndices = False):
    data = []
    # Read the dataset
    file = open(fileName)
    reader = csv.reader(file, delimiter = ',')
    next(reader)
    for row in reader:
        newRow = [float(val) if val else 0 for val in row]
        data.append(newRow)
    file.close()

    n = len(data) # number of observations
    f = len(data[0])
    X = np.array([x[1:f] for x in data]).astype(float)
    #X = np.array([x[21:31] for x in data]).astype(float)
    y = np.array([x[f-1] for x in data]).astype(np.int) #labels

    del data # free up the memory

    if split:
        if returnIndices:
            X_train, X_test, y_train, y_test, indices_train, indices_test =  \
                train_test_split(X, y, range(n), test_size=0.3,
                                 stratify = y, random_state = 7)
            reg_scaler = prep.StandardScaler().fit(X_train)
            X_train = reg_scaler.transform(X_train)
            X_test = reg_scaler.transform(X_test)
            return  X_train, X_test, y_train, y_test, indices_train, indices_test
        else:
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=0.3, stratify = y, random_state = 7)
            reg_scaler = prep.StandardScaler().fit(X_train)
            X_train = reg_scaler.transform(X_train)
            X_test = reg_scaler.transform(X_test)
            return X_train, X_test, y_train, y_test
    else:
        reg_scaler = prep.StandardScaler().fit(X)
        X = reg_scaler.transform(X)
        return X, y


#X, y = loadDataset("data/TwoLeadECG.csv")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = loadDataset("data/TwoLeadECG.csv", split= True)
    svmObj = svm.SVC(C = 10, gamma = .01)
    svmObj.fit(X_train, y_train)

    y_predict = svmObj.predict(X_test)
    #print(y_predict)
    correct = np.sum(y_predict == y_test)
    print("%d out of %d predictions correct" % (correct, len(y_predict)))