#!/home/chenlongzhen/python-miniconda
####################
# use pyFM package  
####################

from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
import numpy as np
## toy example
#train = [
#    {"user": "1", "item": "5", "age": 19},
#    {"user": "2", "item": "43", "age": 33},
#    {"user": "3", "item": "20", "age": 55},
#    {"user": "4", "item": "10", "age": 20},
#]
#v = DictVectorizer()
#X = v.fit_transform(train)
#print(X.toarray())
##[[ 19.   0.   0.   0.   1.   1.   0.   0.   0.]
## [ 33.   0.   0.   1.   0.   0.   1.   0.   0.]
## [ 55.   0.   1.   0.   0.   0.   0.   1.   0.]
## [ 20.   1.   0.   0.   0.   0.   0.   0.   1.]]
#y = np.repeat(1.0,X.shape[0])
#fm = pylibfm.FM()
#fm.fit(X,y)
#fm.predict(v.transform({"user": "1", "item": "10", "age": 24}))
#
## 1, rating example
##import numpy as np
##from sklearn.feature_extraction import DictVectorizer
##from pyfm import pylibfm
#
## Read in data
#def loadData(filename,path="ml-100k/"):
#    data = []
#    y = []
#    users=set()
#    items=set()
#    with open(path+filename) as f:
#        for line in f:
#            (user,movieid,rating,ts)=line.split('\t')
#            data.append({ "user_id": str(user), "movie_id": str(movieid)})
#            y.append(float(rating))
#            users.add(user)
#            items.add(movieid)
#
#    return (data, np.array(y), users, items)
#
#(train_data, y_train, train_users, train_items) = loadData("ua.base")
#(test_data, y_test, test_users, test_items) = loadData("ua.test")
#v = DictVectorizer()
#X_train = v.fit_transform(train_data)
#X_test = v.transform(test_data)
#
## Build and train a Factorization Machine
#fm = pylibfm.FM(num_factors=10, num_iter=100, verbose=True, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal")
#fm.fit(X_train,y_train)
#
#
## Evaluate
#preds = fm.predict(X_test)
#from sklearn.metrics import mean_squared_error
#print("FM MSE: %.4f" % mean_squared_error(y_test,preds)) 

## classification example 
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
X,y = make_classification(n_samples=1000,n_features=100, n_clusters_per_class=1)
#print X[:10]
data = [ {v: k for k, v in dict(zip(i, range(len(i)))).items()}  for i in X]

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1, random_state=42)

v = DictVectorizer()
X_train = v.fit_transform(X_train)
X_test = v.transform(X_test)

fm = pylibfm.FM(num_factors=50, num_iter=10, verbose=True, task="classification", initial_learning_rate=0.0001, learning_rate_schedule="optimal")

fm.fit(X_train,y_train)
# Evaluate
from sklearn.metrics import log_loss
print fm.predict(X_test)
print "Validation log loss: %.4f" % log_loss(y_test,fm.predict(X_test))
