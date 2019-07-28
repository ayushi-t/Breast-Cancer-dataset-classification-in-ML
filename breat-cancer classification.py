#using breast cancer data set
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

df=pd.read_csv('breast-cancer-wisconsin.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'],1, inplace=True)
df.head()
X=np.array(df.drop(['class'], 1))
y=np.array(df['class'])
X
X_train, X_test, y_train, y_test= model_selection.train_test_split(X, y, test_size=0.2)
clf=neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy=clf.score(X_test, y_test)
print(accuracy)
example_measures=np.array([4,1,1,1,2,1,3,1,1])
example_measures=example_measures.reshape(1,-1)
prediction=clf.predict(example_measures)
print(prediction)
example_measures=np.array([[4,1,1,1,2,1,3,1,1], [4,1,1,1,2,1,3,1,1]])
example_measures=example_measures.reshape(2,-1)
prediction=clf.predict(example_measures)
print(prediction)