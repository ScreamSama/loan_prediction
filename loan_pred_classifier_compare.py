# K-Nearest Neighbors (K-NN),Naive_Bayes,Decision_Tree,Random_Forest

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Loan_Prediction.csv')

def valuation_formula(x):
    if x == '3+':
       return 3
    else:
       return x

#Data Preproccesing start

dataset['Dependents'] = dataset.apply(lambda row: valuation_formula(row['Dependents']), axis=1)


X = dataset.iloc[:, 1:12].values
Y = dataset.iloc[:, 12].values


from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

data = X

X = pd.DataFrame(data)
xt = DataFrameImputer().fit_transform(X)

X = xt


# Data preproccesing end

# Encoding 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X0 = LabelEncoder()
X[0] = labelencoder_X0.fit_transform(X[0])
labelencoder_X1 = LabelEncoder()
X[1] = labelencoder_X1.fit_transform(X[1])
labelencoder_X3 = LabelEncoder()
X[3] = labelencoder_X3.fit_transform(X[3])
labelencoder_X4 = LabelEncoder()
X[4] = labelencoder_X4.fit_transform(X[4])
labelencoder_X10 = LabelEncoder()
X[10] = labelencoder_X10.fit_transform(X[10])
onehotencoder = OneHotEncoder(categorical_features = [10])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)


# avoiding dummy variable trap
X = X[:, 1:]

# end of encoding
# feature selection trial
#from sklearn.datasets import load_iris
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
#
#iris = load_iris()
#X, Y = iris.data, iris.target
#X_new = SelectKBest(chi2, k=2).fit_transform(X, Y)
#
#X = X_new

# feature selection trial end

# Splitting data into train and test sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, y_train)

# Naive Bayes 
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dt_classifier.fit(X_train, y_train)

# Random Forest 
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train, y_train)

#KNN prediction
knn_y_pred = knn_classifier.predict(X_test)
#Naive Bayes Prediction
nb_y_pred = nb_classifier.predict(X_test)
#Decision Tree Prediction
dt_y_pred = dt_classifier.predict(X_test)
#Random Forest Prediction
rf_y_pred = rf_classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, knn_y_pred)
cm_nb = confusion_matrix(y_test, nb_y_pred)
cm_dt = confusion_matrix(y_test, dt_y_pred)
cm_rf = confusion_matrix(y_test, rf_y_pred)
